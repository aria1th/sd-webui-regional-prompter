import math
from pprint import pprint
import ldm.modules.attention as ldm_attention_module
import torch
import torchvision
import torchvision.transforms.functional as F
from torchvision.transforms import InterpolationMode, Resize  # Mask.

TOKENSCON = 77
TOKENS = 75

def debug_text(self,text):
    if self.debug:
        print(text)

def main_forward(module ,x, context, # vanilla args
                 mask, divide, # 
                 isvanilla = False,
                 userpp = False,
                 tokens=[],
                 width = 64,
                 height = 64,
                 step = 0, 
                 isxl = False, 
                 negpip = None,
                 regional_prompt_context = None
                 ):
    
    # Forward.
    if regional_prompt_context is None:
        regional_prompt_context = PROMPT_CONTEXT

    if negpip:
        assert len(negpip) == 2, "negpip must be a list of 2 elements but got {}".format(len(negpip))
        conds, cond_tokens = negpip
        context = torch.cat((context,conds),1) # add conds to context with dim 1

    h = module.heads
    if isvanilla: # SBM Ddim / plms have the context split ahead along with x.
        pass
    else: # SBM I think divide may be redundant.
        h = h // divide
    q = module.to_q(x) # attention queries.

    context = ldm_attention_module.default(context, x) # get context from x if not provided.
    k = module.to_k(context) # attention keys.
    v = module.to_v(context) # attention values.

    q, k, v = map(lambda t: ldm_attention_module.rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v)) # cast to attention shape.

    sim = ldm_attention_module.einsum('b i d, b j d -> b i j', q, k) * module.scale # attention similarity.

    if negpip: # negative control
        conds, cond_tokens = negpip
        if cond_tokens:
            for contoken in cond_tokens:
                start = (v.shape[1]//77 - len(cond_tokens)) * 77 # vanilla only accepts 77 tokens at max, so it has to be split
                v[:,start+1:start+contoken,:] = -v[:,start+1:start+contoken,:]  # invert the values corresponding to the tokens, which will be 'negatively regulated'

    if ldm_attention_module.exists(mask): # mask out attention.
        mask = ldm_attention_module.rearrange(mask, 'b ... -> b (...)') # if mask was already in attention shape, cast to batch shape.
        max_neg_value = -torch.finfo(sim.dtype).max # get the lowest possible value for the attention.
        mask = ldm_attention_module.repeat(mask, 'b j -> (b h) () j', h=h) # repeat mask h times, which is required for matching h heads.
        sim.masked_fill_(~mask, max_neg_value) # mask out the attention similarity with the lowest possible value.

    attn = sim.softmax(dim=-1) # calculate attention with softmax.

    ## for prompt mode make basemask from attention maps
    hiresscaler(height,width,attn, context=regional_prompt_context)

    if userpp and step > 0:
        for b in range(attn.shape[0] // 8):
            if regional_prompt_context.masks_hw == []:
                regional_prompt_context.masks_hw = [(height,width)]
            elif (height,width) not in regional_prompt_context.masks_hw:
                regional_prompt_context.masks_hw.append((height,width))

            for t in tokens:
                power = 4 if isxl else 1.2
                add = attn[8*b:8*(b+1),:,t[0]:t[0]+len(t)]**power
                add = torch.sum(add,dim = 2)
                t = f"{t}-{b}"         
                if t not in regional_prompt_context.masks:
                    regional_prompt_context.masks[t] = add
                else:
                    if regional_prompt_context.masks[t].shape[1] != add.shape[1]:
                        add = add.view(8,height,width)
                        add = F.resize(add,regional_prompt_context.masks_hw[0])
                        add = add.reshape_as(regional_prompt_context.masks[t])

                    regional_prompt_context.masks[t] = regional_prompt_context.masks[t] + add

    out = ldm_attention_module.einsum('b i j, b j d -> b i d', attn, v)
    out = ldm_attention_module.rearrange(out, '(b h) n d -> b n (h d)', h=h)
    out = module.to_out(out)

    return out

def hook_forwards(self, root_module: torch.nn.Module, remove=False, regional_prompt_context = None):
    self.hooked = True if not remove else False
    for name, module in root_module.named_modules():
        if "attn2" in name and module.__class__.__name__ == "CrossAttention":
            module.forward = hook_forward(self, module, regional_prompt_context = regional_prompt_context)
            if remove:
                del module.forward

################################################################################
##### Attention mode 

def hook_forward(self, module, regional_prompt_context = None):
    if regional_prompt_context is None:
        regional_prompt_context = PROMPT_CONTEXT
    def forward(x, context=None, mask=None, additional_tokens=None, n_times_crossframe_attn_in_self=0):
        # regional_prompt_context is referenced at function creation time, so it's not a default argument.
        if self.debug :
            print("input : ", x.size())
            print("tokens : ", context.size())
            print("module : ", getattr(module, self.layer_name,None))

        if self.xsize == 0: self.xsize = x.shape[1]
        if "input" in getattr(module, self.layer_name,""):
            if x.shape[1] > self.xsize:
                self.in_hr = True

        height = self.hr_h if self.in_hr and self.hr else self.h 
        width = self.hr_w if self.in_hr and self.hr else self.w

        xs = x.size()[1]
        scale = round(math.sqrt(height * width / xs))

        dsh = round(height / scale)
        dsw = round(width / scale)
        ha, wa = xs % dsh, xs % dsw
        if ha == 0:
            dsw = int(xs / dsh)
        elif wa == 0:
            dsh = int(xs / dsw)

        contexts = context.clone()

        # SBM Matrix mode.
        def matsepcalc(x,contexts,mask,pn,divide):
            h_states = []
            xs = x.size()[1]
            (dsh,dsw) = split_dims(xs, height, width, self)
            
            if "Horizontal" in self.mode: # Map columns / rows first to outer / inner.
                dsout = dsw
                dsin = dsh
            elif "Vertical" in self.mode:
                dsout = dsh
                dsin = dsw

            tll = self.pt if pn else self.nt
            
            i = 0
            outb = None
            if self.usebase:
                context = contexts[:,tll[i][0] * TOKENSCON:tll[i][1] * TOKENSCON,:]
                # SBM Controlnet sends extra conds at the end of context, apply it to all regions.
                cnet_ext = contexts.shape[1] - (contexts.shape[1] // TOKENSCON) * TOKENSCON
                if cnet_ext > 0:
                    context = torch.cat([context,contexts[:,-cnet_ext:,:]],dim = 1)
                    
                negpip = negpipdealer(i,pn)

                i = i + 1

                out = main_forward(module, x, context, mask, divide, self.isvanilla,userpp =True,step = self.step, isxl = self.isxl, negpip = negpip,
                                   regional_prompt_context=regional_prompt_context)

                if len(self.nt) == 1 and not pn:
                    debug_text(self,"return out for NP")
                    return out
                # if self.usebase:
                outb = out.clone()
                outb = outb.reshape(outb.size()[0], dsh, dsw, outb.size()[2]) if "Ran" not in self.mode else outb

            sumout = 0
            debug_text(self,f"tokens : {tll},pn : {pn}")
            debug_text(self,[r for r in self.aratios])

            for drow in self.aratios:
                v_states = []
                sumin = 0
                for dcell in drow.cols:
                    # Grabs a set of tokens depending on number of unrelated breaks.
                    context = contexts[:,tll[i][0] * TOKENSCON:tll[i][1] * TOKENSCON,:]
                    # SBM Controlnet sends extra conds at the end of context, apply it to all regions.
                    cnet_ext = contexts.shape[1] - (contexts.shape[1] // TOKENSCON) * TOKENSCON
                    if cnet_ext > 0:
                        context = torch.cat([context,contexts[:,-cnet_ext:,:]],dim = 1)
                        
                    negpip = negpipdealer(i,pn)

                    debug_text(self,f"tokens : {tll[i][0]*TOKENSCON}-{tll[i][1]*TOKENSCON}")
                    i = i + 1 + dcell.breaks
                    # if i >= contexts.size()[1]: 
                    #     indlast = True

                    out = main_forward(module, x, context, mask, divide, self.isvanilla,userpp = self.pn, step = self.step, isxl = self.isxl,negpip = negpip,
                                       regional_prompt_context=regional_prompt_context)
                    debug_text(self,f" dcell.breaks : {dcell.breaks}, dcell.ed : {dcell.ed}, dcell.st : {dcell.st}")
                    if len(self.nt) == 1 and not pn:
                        debug_text(self,"return out for NP")
                        return out
                    # Actual matrix split by region.
                    if "Ran" in self.mode:
                        v_states.append(out)
                        continue
                    
                    out = out.reshape(out.size()[0], dsh, dsw, out.size()[2]) # convert to main shape.
                    # if indlast:
                    addout = 0
                    addin = 0
                    sumin = sumin + int(dsin*dcell.ed) - int(dsin*dcell.st)
                    if dcell.ed >= 0.999:
                        addin = sumin - dsin
                        sumout = sumout + int(dsout*drow.ed) - int(dsout*drow.st)
                        if drow.ed >= 0.999:
                            addout = sumout - dsout
                    if "Horizontal" in self.mode:
                        out = out[:,int(dsh*drow.st) + addout:int(dsh*drow.ed),
                                    int(dsw*dcell.st) + addin:int(dsw*dcell.ed),:]
                        if self.debug : print(f"{int(dsh*drow.st) + addout}:{int(dsh*drow.ed)},{int(dsw*dcell.st) + addin}:{int(dsw*dcell.ed)}")
                        if self.usebase : 
                            # outb_t = outb[:,:,int(dsw*drow.st):int(dsw*drow.ed),:].clone()
                            outb_t = outb[:,int(dsh*drow.st) + addout:int(dsh*drow.ed),
                                            int(dsw*dcell.st) + addin:int(dsw*dcell.ed),:].clone()
                            out = out * (1 - dcell.base) + outb_t * dcell.base
                    elif "Vertical" in self.mode: # Cols are the outer list, rows are cells.
                        out = out[:,int(dsh*dcell.st) + addin:int(dsh*dcell.ed),
                                  int(dsw*drow.st) + addout:int(dsw*drow.ed),:]
                        debug_text(self,f"{int(dsh*dcell.st) + addin}:{int(dsh*dcell.ed)}-{int(dsw*drow.st) + addout}:{int(dsw*drow.ed)}")
                        if self.usebase : 
                            # outb_t = outb[:,:,int(dsw*drow.st):int(dsw*drow.ed),:].clone()
                            outb_t = outb[:,int(dsh*dcell.st) + addin:int(dsh*dcell.ed),
                                          int(dsw*drow.st) + addout:int(dsw*drow.ed),:].clone()
                            out = out * (1 - dcell.base) + outb_t * dcell.base
                    debug_text(self,f"sumin:{sumin},sumout:{sumout},dsh:{dsh},dsw:{dsw}")
            
                    v_states.append(out)
                    if self.debug : 
                        for h in v_states:
                            print(h.size())
                            
                if "Horizontal" in self.mode:
                    ox = torch.cat(v_states,dim = 2) # First concat the cells to rows.
                elif "Vertical" in self.mode:
                    ox = torch.cat(v_states,dim = 1) # Cols first mode, concat to cols.
                elif "Ran" in self.mode:
                    if self.usebase:
                        ox = outb * makerrandman(self.ranbase,dsh,dsw).view(-1, 1)
                    ox = torch.zeros_like(v_states[0])
                    for state, filter in zip(v_states, self.ransors):
                        filter = makerrandman(filter,dsh,dsw)
                        ox = ox + state * filter.view(-1, 1)
                    return ox

                h_states.append(ox)
            if "Horizontal" in self.mode:
                ox = torch.cat(h_states,dim = 1) # Second, concat rows to layer.
            elif "Vertical" in self.mode:
                ox = torch.cat(h_states,dim = 2) # Or cols.
            ox = ox.reshape(x.size()[0],x.size()[1],x.size()[2]) # Restore to 3d source.  
            return ox

        def masksepcalc(x,contexts,mask,pn,divide):
            xs = x.size()[1]
            (dsh,dsw) = split_dims(xs, height, width, self)

            tll = self.pt if pn else self.nt
            
            # Base forward.
            i = 0
            outb = None
            if self.usebase:
                context = contexts[:,tll[i][0] * TOKENSCON:tll[i][1] * TOKENSCON,:]
                # SBM Controlnet sends extra conds at the end of context, apply it to all regions.
                cnet_ext = contexts.shape[1] - (contexts.shape[1] // TOKENSCON) * TOKENSCON
                if cnet_ext > 0:
                    context = torch.cat([context,contexts[:,-cnet_ext:,:]],dim = 1)

                negpip = negpipdealer(i,pn) 

                i = i + 1
                out = main_forward(module, x, context, mask, divide, self.isvanilla, isxl = self.isxl, negpip = negpip,
                                   regional_prompt_context=regional_prompt_context)

                if len(self.nt) == 1 and not pn:
                    debug_text(self,"return out for NP")
                    return out
                # if self.usebase:
                outb = out.clone()
                outb = outb.reshape(outb.size()[0], dsh, dsw, outb.size()[2]) 

            debug_text(self,f"tokens : {tll},pn : {pn}")
            
            ox = torch.zeros_like(x)
            ox = ox.reshape(ox.shape[0], dsh, dsw, ox.shape[2])
            ftrans = Resize((dsh, dsw), interpolation = InterpolationMode("nearest"))
            for rmask in self.regmasks:
                # Need to delay mask tensoring so it's on the correct gpu.
                # Dunno if caching masks would be an improvement.
                if self.usebase:
                    bweight = self.bratios[0][i - 1]
                # Resize mask to current dims.
                # Since it's a mask, we prefer a binary value, nearest is the only option.
                rmask2 = ftrans(rmask.reshape([1, *rmask.shape])) # Requires dimensions N,C,{d}.
                rmask2 = rmask2.reshape(1, dsh, dsw, 1)
                
                # Grabs a set of tokens depending on number of unrelated breaks.
                context = contexts[:,tll[i][0] * TOKENSCON:tll[i][1] * TOKENSCON,:]
                # SBM Controlnet sends extra conds at the end of context, apply it to all regions.
                cnet_ext = contexts.shape[1] - (contexts.shape[1] // TOKENSCON) * TOKENSCON
                if cnet_ext > 0:
                    context = torch.cat([context,contexts[:,-cnet_ext:,:]],dim = 1)
                    
                debug_text(self,f"tokens : {tll[i][0]*TOKENSCON}-{tll[i][1]*TOKENSCON}")
                i = i + 1
                # if i >= contexts.size()[1]: 
                #     indlast = True
                out = main_forward(module, x, context, mask, divide, self.isvanilla, isxl = self.isxl,
                                   regional_prompt_context=regional_prompt_context)
                if len(self.nt) == 1 and not pn:
                    debug_text(self,"return out for NP")
                    return out
                    
                out = out.reshape(out.size()[0], dsh, dsw, out.size()[2]) # convert to main shape.
                if self.usebase:
                    out = out * (1 - bweight) + outb * bweight
                ox = ox + out * rmask2

            if self.usebase:
                rmask = self.regbase
                rmask2 = ftrans(rmask.reshape([1, *rmask.shape])) # Requires dimensions N,C,{d}.
                rmask2 = rmask2.reshape(1, dsh, dsw, 1)
                ox = ox + outb * rmask2
            ox = ox.reshape(x.size()[0],x.size()[1],x.size()[2]) # Restore to 3d source.  
            return ox

        def promptsepcalc(x, contexts, mask, pn,divide, regional_prompt_context = None):
            if regional_prompt_context is None:
                regional_prompt_context = PROMPT_CONTEXT
            h_states = []

            tll = self.pt if pn else self.nt
            debug_text(self,f"tokens : {tll},pn : {pn}")

            for i, tl in enumerate(tll):
                context = contexts[:, tl[0] * TOKENSCON : tl[1] * TOKENSCON, :]
                # SBM Controlnet sends extra conds at the end of context, apply it to all regions.
                cnet_ext = contexts.shape[1] - (contexts.shape[1] // TOKENSCON) * TOKENSCON
                if cnet_ext > 0:
                    context = torch.cat([context,contexts[:,-cnet_ext:,:]],dim = 1)
                
                debug_text(self,f"tokens : {tl[0]*TOKENSCON}-{tl[1]*TOKENSCON}")

                userpp = self.pn and i == 0 and self.pfirst

                negpip = negpipdealer(self.condi,pn) if "La" in self.calc else negpipdealer(i,pn)

                out = main_forward(module, x, context, mask, divide, self.isvanilla, 
                                   userpp = userpp, width = dsw, height = dsh, tokens = self.pe, 
                                   step = self.step, isxl = self.isxl, negpip = negpip,
                                   regional_prompt_context=regional_prompt_context)

                if (len(self.nt) == 1 and not pn) or ("Pro" in self.mode and "La" in self.calc):
                    debug_text(self,"return out for NP or Latent")
                    return out

                debug_text(self,[scale, dsh, dsw, dsh * dsw, x.size()[1]])

                if i == 0:
                    outb = out.clone()
                    continue
                else:
                    h_states.append(out)

            if self.debug:
                for h in h_states :
                    print(f"divided : {h.size()}")
                print(regional_prompt_context.masks_hw)

            if regional_prompt_context.masks_hw == []:
                return outb

            ox = outb.clone() if self.ex else outb * 0

            debug_text(self,[regional_prompt_context.masks_hw,regional_prompt_context.is_mask_ready,(dsh,dsw) in regional_prompt_context.masks_hw and regional_prompt_context.is_mask_ready,len(regional_prompt_context.masks_f),len(h_states)])

            if (dsh,dsw) in regional_prompt_context.masks_hw and regional_prompt_context.is_mask_ready:
                depth = regional_prompt_context.masks_hw.index((dsh,dsw))
                maskb = None
                for masks , state in zip(regional_prompt_context.masks_f.values(),h_states):
                    mask = masks[depth]
                    masked = torch.multiply(state, mask)
                    if self.ex:
                        ox = torch.where(masked !=0 , masked, ox)
                    else:
                        ox = ox + masked
                    maskb = maskb + mask if maskb is not None else mask
                maskb = 1 - maskb
                if not self.ex : ox = ox + torch.multiply(outb, maskb)
                return ox
            else:
                return outb

        if self.eq:
            debug_text(self,"same token size and divisions")
            if "Mas" in self.mode:
                ox = masksepcalc(x, contexts, mask, True, 1)
            elif "Pro" in self.mode:
                ox = promptsepcalc(x, contexts, mask, True, 1, regional_prompt_context=regional_prompt_context)
            else:
                ox = matsepcalc(x, contexts, mask, True, 1)
        elif x.size()[0] == 1 * self.batch_size:
            debug_text(self,"different tokens size")
            if "Mas" in self.mode:
                ox = masksepcalc(x, contexts, mask, self.pn, 1)
            elif "Pro" in self.mode:
                ox = promptsepcalc(x, contexts, mask, self.pn, 1, regional_prompt_context=regional_prompt_context)
            else:
                ox = matsepcalc(x, contexts, mask, self.pn, 1)
        else:
            debug_text(self,"same token size and different divisions")
            # SBM You get 2 layers of x, context for pos/neg.
            # Each should be forwarded separately, pairing them up together.
            if self.isvanilla: # SBM Ddim reverses cond/uncond.
                nx, px = x.chunk(2)
                conn,conp = contexts.chunk(2)
            else:
                px, nx = x.chunk(2)
                conp,conn = contexts.chunk(2)
            if "Mas" in self.mode:
                opx = masksepcalc(px, conp, mask, True, 2)
                onx = masksepcalc(nx, conn, mask, False, 2)
            elif "Pro" in self.mode:
                opx = promptsepcalc(px, conp, mask, True, 2, regional_prompt_context=regional_prompt_context)
                onx = promptsepcalc(nx, conn, mask, False, 2, regional_prompt_context=regional_prompt_context)
            else:
                # SBM I think division may have been an incorrect patch.
                # But I'm not sure, haven't tested beyond DDIM / PLMS.
                opx = matsepcalc(px, conp, mask, True, 2)
                onx = matsepcalc(nx, conn, mask, False, 2)
            if self.isvanilla: # SBM Ddim reverses cond/uncond.
                ox = torch.cat([onx, opx])
            else:
                ox = torch.cat([opx, onx])  

        self.count += 1

        limit = 70 if self.isxl else 16

        if self.count == limit:
            self.pn = not self.pn
            self.count = 0
            self.pfirst = False
            self.condi += 1
        debug_text(self,f"output : {ox.size()}")
        return ox

    return forward

def split_dims(xs, height, width, self):
    """Split an attention layer dimension to height + width.
    
    Originally, the estimate was dsh = sqrt(hw_ratio*xs),
    rounding to the nearest value. But this proved inaccurate.
    What seems to be the actual operation is as follows:
    - Divide h,w by 8, rounding DOWN. 
      (However, webui forces dims to be divisible by 8 unless set explicitly.)
    - For every new layer (of 4), divide both by 2 and round UP (then back up)
    - Multiply h*w to yield xs.
    There is no inverse function to this set of operations,
    so instead we mimic them sans the multiplication part with orig h+w.
    The only alternative is brute forcing integer guesses,
    which might be inaccurate too.
    No known checkpoints follow a different system of layering,
    but it's theoretically possible. Please report if encountered.
    """
    # OLD METHOD.
    # scale = round(math.sqrt(height*width/xs))
    # dsh = round_dim(height, scale)
    # dsw = round_dim(width, scale) 
    scale = math.ceil(math.log2(math.sqrt(height * width / xs)))
    dsh = repeat_div(height,scale)
    dsw = repeat_div(width,scale)
    if xs > dsh * dsw and hasattr(self,"nei_multi"):
        dsh, dsw = self.nei_multi[1], self.nei_multi[0] 
        while dsh*dsw != xs:
            dsh, dsw = dsh//2, dsw//2

    if self.debug : print(scale,dsh,dsw,dsh*dsw,xs, height, width)

    return dsh,dsw

def repeat_div(x,y):
    """Imitates dimension halving common in convolution operations.
    
    This is a pretty big assumption of the model,
    but then if some model doesn't work like that it will be easy to spot.
    """
    while y > 0:
        x = math.ceil(x / 2)
        y = y - 1
    return x

#################################################################################
##### for Prompt mode
class PromptProcessingContext:
    def __init__(self):
        self.masks = []
        self.masks_hw = []
        self.masks_f = []
        self.is_mask_ready = False
        
    def reset(self):
        self.masks.clear()
        self.masks_hw.clear()
        self.masks_f.clear()
        self.is_mask_ready = False
        
PROMPT_CONTEXT = PromptProcessingContext()
        
def get_default_context():
    return PROMPT_CONTEXT

def reset_prompt_context(self, context = None): # init parameters in every batch
    if context is None:
        context = PROMPT_CONTEXT
    context.reset()
    self.step = 0
    self.x = None
    self.rebacked = False

def save_prompt_context(self, processed, context = None): # save masks in every batch
    if context is None:
        context = PROMPT_CONTEXT
    for mask ,th in zip(context.masks.values(),self.th):
        img, _ , _= makepmask(mask, self.h, self.w,th, self.step, context=context)
        processed.images.append(img)
    return processed

def hiresscaler(height, width, attn, context = None):
    """
    If masks_hw is not empty, resize masks to the largest height and width.
    """
    if context is None:
        context = PROMPT_CONTEXT
    if context.masks_hw != []:
        if height > context.masks_hw[0][0]: # [0][0] has largest height, if in hires, height will be larger than [0][0]
            (oh, ow) = context.masks_hw[0]
            del context.masks_hw
            context.masks_hw = [(height,width)]
            hiresmask(context.masks,oh, ow, height, width,attn[:,:,0])
            for i in range(4):
                m = (2 ** (i))
                hiresmask(context.masks_f,oh//m, ow//m, height//m,width//m ,torch.zeros(1,height*width //m**2,1),i = i )

def hiresmask(masks,oh,ow,nh,nw,at,i = None):
    """
    In-place resize of masks.
    """
    for key in masks.keys():
        mask = masks[key] if i is None else masks[key][i]
        mask = mask.view(8 if i is None else 1,oh,ow)
        mask = F.resize(mask,(nh,nw))
        mask = mask.reshape_as(at)
        if i is None:
            masks[key] = mask
        else:
            masks[key][i] = mask

def makepmask(mask, h, w, th, step, bratio = 1, context=None): # make masks from attention cache return [for preview, for attention, for Latent]
    """
    Generate a mask then return it as triple.
    """
    if context is None:
        context = PROMPT_CONTEXT
    th = th - step * 0.005
    bratio = 1 - bratio
    mask = torch.mean(mask,dim=0)
    mask = mask / mask.max().item()
    mask = torch.where(mask > th ,1,0)
    mask = mask.float()
    mask = mask.view(1,context.masks_hw[0][0],context.masks_hw[0][1]) 
    img = torchvision.transforms.functional.to_pil_image(mask)
    img = img.resize((w,h))
    mask = F.resize(mask,(h,w),interpolation=F.InterpolationMode.NEAREST)
    lmask = mask
    mask = mask.reshape(h*w)
    mask = torch.where(mask > 0.1 ,1,0)
    return img,mask * bratio , lmask * bratio

def makerrandman(mask, h, w, latent = False): # make masks from attention cache return [for preview, for attention, for Latent]
    """
    Generate a mask from a random tensor.
    """
    mask = mask.float()
    mask = mask.view(1,mask.shape[0],mask.shape[1]) 
    img = torchvision.transforms.functional.to_pil_image(mask)
    img = img.resize((w,h))
    mask = F.resize(mask,(h,w),interpolation=F.InterpolationMode.NEAREST)
    if latent: return mask
    mask = mask.reshape(h*w)
    mask = torch.round(mask).long()
    return mask

def negpipdealer(i,pn):
    negpip = None
    from modules.scripts import scripts_txt2img
    for script in scripts_txt2img.alwayson_scripts:
        if "negpip.py" in script.filename:
            negpip = script

    if negpip:
        conds = negpip.conds if pn else negpip.unconds
        tokens = negpip.cond_tokens if pn else negpip.untokens
        if conds and len(conds) >= i + 1:
            if conds[i] is not None:
                return [conds[i],tokens[i]]
        else:
            return None
    else:
        return None