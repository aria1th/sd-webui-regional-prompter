# attaches API for returning detected color mask from image
from scripts.regions import draw_image, inpaintmaskdealer
from scripts.contexts import RegionsProcessingContext, FakeMaskDebuggingContext

def get_image_mask(image, usebase=False):
    """
    Returns a color mask from an image.
    """
    context = RegionsProcessingContext()
    polymask_image = draw_image(image, context)
    debug_obj = FakeMaskDebuggingContext()
    # usebase : bool
    debug_obj.usebase = usebase
    inpaintmaskdealer(
        debug_obj,
        "",
        0.2, False, 
        polymask_image
    )
    # in-place logged at debug_obj
    debug_obj.regbase #tensor if usebase else None, outer regions
    debug_obj.regmasks
    
    
    
    