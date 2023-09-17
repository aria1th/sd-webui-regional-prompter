class RegionsProcessingContext:
    def __init__(self):
        self.state_dict = {
            "COLREG": None, # Computed colour regions cache. Array. Extended whenever a new colour is requested.
            "REGUSE": dict(), # Used regions. Reset on new canvas / upload (preset).
            "VARIANT": 0, # Ensures that the sketch canvas is actually refreshed.
            "INDCOLREPL": False,
        }
        
    # set accessors
    def __setitem__(self, key, value):
        self.state_dict[key] = value
        
    # get accessors
    def __getitem__(self, key):
        return self.state_dict[key]
    
    # hint accessors
    def keys_hint(self):
        return ("COLREG","REGUSE","VARIANT","INDCOLREPL")
    
    # update from same object
    def update(self, other:"RegionsProcessingContext"):
        self.state_dict.update(other.state_dict)
        
    def debug(self):
        print(self.state_dict)
    
class FakeMaskDebuggingContext:
    """
    Fake object for debugging.
    """
    def __init__(self) -> None:
        self.debug = False
        self.regmasks = []
        self.usebase = False
        self.regbase = None
        
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