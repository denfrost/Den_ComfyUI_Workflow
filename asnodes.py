import torch
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import sys, os
import comfy.model_management


MAX_RESOLUTION = 8192

#Latent space from size Image
class Den_ImageToLatentSpace:
    @classmethod
    def INPUT_TYPES(s):
        return \
        {"required":
            {
            "image": ("IMAGE",),
            "batch_size": ("INT", {"default": 1, "min": 1, "max": 4096})
            }
        }

    RETURN_TYPES = ("LATENT", "image")
    RETURN_TYPES = ("LATENT", "IMAGE")
    FUNCTION = "generate"

    CATEGORY = "latent"
    def generate(self, image, batch_size=1):
        height = image.shape[1]
        width = image.shape[2]
        latent = torch.zeros([batch_size, 4, height // 8, width // 8], device=comfy.model_management.intermediate_device())
        return ({"samples":latent}, image,)


class Den_MaskToImage_AS:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"mask": ("MASK",),}}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "convert"
    CATEGORY = "ASNodes"

    def convert(self, mask):
        d2, d3 = mask.size()
        print("MASK SIZE:", mask.size())

        new_image = torch.zeros(
            (1, d2, d3, 3),
            dtype=torch.float32,
        )
        new_image[0, :, :, 0] = mask
        new_image[0, :, :, 1] = mask
        new_image[0, :, :, 2] = mask

        print("MaskSize", mask.size())
        print("Tyep New img", type(new_image))

        return (new_image,)


class Den_ImageToMask_AS:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"image": ("IMAGE",),}}

    RETURN_TYPES = ("MASK",)
    FUNCTION = "convert"
    CATEGORY = "ASNodes"

    def convert(self, image):
        return (image.squeeze().mean(2),)


class Den_LatentMix_AS:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "samples_to": ("LATENT",),
                              "samples_from": ("LATENT",),
                              "blend": ("FLOAT", {"default": 0, "min": 0, "max": 100, "step": 1}),
                            }}
    
    RETURN_TYPES = ("LATENT",)
    FUNCTION = "composite"
    CATEGORY = "ASNodes"

    def composite(self, samples_to, samples_from, blend):
        samples_out = samples_to.copy()
        s_to = samples_to["samples"].clone()
        s_from = samples_from["samples"].clone()
        samples_out["samples"] = s_to * blend / 100 + s_from * (100 - blend) / 100
        return (samples_out,)


class Den_LatentAdd_AS:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "samples_to": ("LATENT",),
                              "samples_from": ("LATENT",),
                            }}
    
    RETURN_TYPES = ("LATENT",)
    FUNCTION = "composite"
    CATEGORY = "ASNodes"

    def composite(self, samples_to, samples_from):
        samples_out = samples_to.copy()
        s_to = samples_to["samples"].clone()
        s_from = samples_from["samples"].clone()
        samples_out["samples"] = (s_to + s_from)
        return (samples_out,)


class Den_SaveLatent_AS:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "latent_in": ("LATENT",), }}
    
    RETURN_TYPES = ("LATENT",)
    FUNCTION = "doStuff"
    CATEGORY = "ASNodes"

    def doStuff(self, latent_in):
        torch.save(latent_in, 'latent.pt')
        return (latent_in, )
    

# a = torch.load("e:/portables/ComfyUI_windows_portable/latent.pt")   
# for idx in range(a['samples'].shape[1]):
#     plt.figure()
#     plt.imshow(a['samples'][0,idx,:,:])
# plt.show()

class Den_LoadLatent_AS:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {}}
    
    RETURN_TYPES = ("LATENT",)
    FUNCTION = "doStuff"
    CATEGORY = "ASNodes"

    def doStuff(self,):
        latent_out = torch.load('latent.pt')
        return (latent_out, )
    

class Den_LatentToImages_AS:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "latent_in": ("LATENT",), }}
    
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "doStuff"
    CATEGORY = "ASNodes"

    def doStuff(self, latent_in):
        s = latent_in['samples']
        s = (s - s.min()) / (s.max() - s.min())
        d1, d2, d3, d4 = s.shape
        images_out = torch.zeros(d2, d3, d4, 3)
        
        for idx in range(s.shape[1]):
            for chan in range(3):
                images_out[idx,:,:,chan] = s[0,idx,:,:]
        return (images_out, )


class Den_LatentMixMasked_As:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "samples_to": ("LATENT",),
                              "samples_from": ("LATENT",),
                              "mask": ("MASK",),
                              }}
    
    RETURN_TYPES = ("LATENT",)
    FUNCTION = "composite"
    CATEGORY = "ASNodes"

    def composite(self, samples_to, samples_from, mask):
        print(samples_to["samples"].size())
        samples_out = samples_to.copy()
        s_to = samples_to["samples"].clone()
        s_from = samples_from["samples"].clone()
        samples_out["samples"] = s_to * mask + s_from * (1 - mask)
        return (samples_out,)


class Den_ImageMixMasked_As:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "image_to": ("IMAGE",),
                              "image_from": ("IMAGE",),
                              "mask": ("MASK",),
                              }}
    
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "composite"
    CATEGORY = "ASNodes"

    def composite(self, image_to, image_from, mask):

        image_out = image_to.clone()    
        image_out[0,:,:,0] = image_to[0,:,:,0] * mask + image_from[0,:,:,0] * (1 - mask)
        image_out[0,:,:,1] = image_to[0,:,:,1] * mask + image_from[0,:,:,1] * (1 - mask)
        image_out[0,:,:,2] = image_to[0,:,:,2] * mask + image_from[0,:,:,2] * (1 - mask)
        return (image_out,)


class Den_TextToImage_AS:
    @classmethod
    def INPUT_TYPES(s):
        fonts = os.listdir('C:\Windows\Fonts')

        return {
            "required": {
                "text": ("STRING", {"multiline": True}),
                "font": (fonts, ),
                "size": ("INT", {"default": 20, "min": 1, "max": MAX_RESOLUTION, "step": 1}),
                "width": ("INT", {"default": 512, "min": 64, "max": MAX_RESOLUTION, "step": 64}),
                "height": ("INT", {"default": 512, "min": 64, "max": MAX_RESOLUTION, "step": 64}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "doStuff"
    CATEGORY = "ASNodes"

    def doStuff(self, text, font, size, width, height):
        
        PIL_image = Image.new("RGB", (width, height), (0, 0, 0))
        
        draw = ImageDraw.Draw(PIL_image)

        # Set the font and size
        font = ImageFont.truetype(font, size)

        # Get the size of the text
        text_size = draw.textsize(text, font)

        # Calculate the position of the text
        x = (PIL_image.width - text_size[0]) / 2
        y = (PIL_image.height - text_size[1]) / 2

        # Draw the text on the image
        draw.text((x, y), text, font=font, fill=(255,255,255))

        np_image = np.array(PIL_image)

        new_image = torch.zeros(
            (1, height, width, 3),
            dtype=torch.float32,
        )
        new_image[0,:,:,:] = torch.from_numpy(np_image) / 256
        
        return (new_image,)


class Den_BatchIndex_AS:
    def __init__(self) -> None:
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "batch_index": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            },
        }

    RETURN_TYPES = ("FLOAT",)
    FUNCTION = "doStuff"
    CATEGORY = "ASNodes"

    def doStuff(self, batch_index):
        return (batch_index,)
    

class Den_MapRange_AS:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "value": ("FLOAT", {"default": 0, "min": -sys.float_info.max, "max": sys.float_info.max, "step": 0.01}),
                "in_0": ("FLOAT", {"default": 0, "min": -sys.float_info.max, "max": sys.float_info.max, "step": 0.01}),
                "in_1": ("FLOAT", {"default": 1, "min": -sys.float_info.max, "max": sys.float_info.max, "step": 0.01}),
                "out_0": ("FLOAT", {"default": 0, "min": -sys.float_info.max, "max": sys.float_info.max, "step": 0.01}),
                "out_1": ("FLOAT", {"default": 1, "min": -sys.float_info.max, "max": sys.float_info.max, "step": 0.01}),
                
            },
        }

    RETURN_TYPES = ("FLOAT", "INT")
    FUNCTION = "mapRange"
    CATEGORY = "ASNodes"

    def mapRange(self, value, in_0, in_1, out_0, out_1):
        if (in_0 == in_1):
            raise ValueError("MapRange_AS: in_0 and in_1 are equal")
        
        run_param = (value - in_0) / (in_1 - in_0)
        result = out_0 + run_param * (out_1 - out_0)
        return (result, round(result))


class Den_Number2Float_AS:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "value": ("number", { }),
            },
        }

    RETURN_TYPES = ("FLOAT",)
    FUNCTION = "convert"
    CATEGORY = "ASNodes"

    def convert(self, value):
        return (value, )


class Den_Int2Any_AS:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "value": ("INT", {"default": 0, }),
            },
        }

    RETURN_TYPES = ("*",)
    FUNCTION = "convert"
    CATEGORY = "ASNodes"

    def convert(self, value):
        return (value, )


class Den_Number2Int_AS:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "value": ("number", { }),
            },
        }

    RETURN_TYPES = ("INT",)
    FUNCTION = "convert"
    CATEGORY = "ASNodes"

    def convert(self, value):
        return (round(value), )


class Den_Eval_AS:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "optional": {
                "i1": ("INT", { "default": 0, "step": 1 }),
                "i2": ("INT", { "default": 0, "step": 1 }),
                "f1": ("FLOAT", { "default": 0.0, "step": 0.1, "round": 0.01 }),
                "f2": ("FLOAT", { "default": 0.0, "step": 0.1, "round": 0.01 }),
                "s1": ("STRING", { "multiline": True, "default": "" }),
                "s2": ("STRING", { "multiline": True, "default": "" }),
            },
            "required": {
                "int_prc": ("STRING", { "multiline": False, "default": "0" }),
                "float_prc": ("STRING", { "multiline": False, "default": "0" }),
                "str_prc": ("STRING", { "multiline": False, "default": "0" }),
            },

        }

    RETURN_TYPES = ("INT","FLOAT","STRING",)
    FUNCTION = "do_stuff"
    CATEGORY = "ASNodes"

    def do_stuff(self, int_prc, float_prc, str_prc, i1, i2, f1, f2, s1, s2):
        int_out = eval(int_prc)
        float_out = eval(float_prc)
        str_out = eval(str_prc)
        
        return (int_out, float_out, str_out )


class Den_Number_AS:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "value": ("FLOAT", {"default": 0, "min": -sys.float_info.max, "max": sys.float_info.max, }),
            },
        }

    RETURN_TYPES = ("number",)
    FUNCTION = "doStuff"
    CATEGORY = "ASNodes"

    def doStuff(self, value):
        return (value, )
    

class Den_Math_AS:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "do": (["add", "subtract", "multiply", "divide", "power"], ),
                "in_0": ("FLOAT", {"default": 0, "step": 0.01, "round": 0.01,}),
                "in_1": ("FLOAT", {"default": 0, "step": 0.01, "round": 0.01,}),
            },
        }

    RETURN_TYPES = ("FLOAT", "INT")
    FUNCTION = "calculate"
    CATEGORY = "ASNodes"

    def calculate(self, do, in_0, in_1):
        if do=="add":
            result = in_0 + in_1
        if do=="subtract":
            result = in_0 - in_1
        if do=="multiply":
            result = in_0 * in_1
        if do=="divide":
            result = in_0 / in_1
        if do=="power":
            result = in_0 ** in_1

        return (result, round(result) )
    

class Den_Increment_AS:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "value": ("INT", {"default": 10, "min": 0, "max": 20, "step":3}),
            },
        }

    RETURN_TYPES = ("number",)
    FUNCTION = "doStuff"
    CATEGORY = "ASNodes"

    def doStuff(self, value):
        return (value, )


class Den_CropImage_AS:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "image": ("IMAGE",),
                             "width": ("INT", {"default": 0, "min": 0, "max": MAX_RESOLUTION}),
                             "height": ("INT", {"default": 0, "min": 0, "max": MAX_RESOLUTION}),
                             "x": ("INT", {"default": 0, "min": 0, "max": MAX_RESOLUTION}),
                             "y": ("INT", {"default": 0, "min": 0, "max": MAX_RESOLUTION}),
                         }}
    
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "process"
    CATEGORY = "ASNodes"

    def process(self, image, width, height, x, y):

        image_out = image.clone()   
        print(image.shape) 
        image_out = image_out[:, y:y+height, x:x+width]
        return (image_out,)
    

class Den_TextWildcardList_AS:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"text": ("STRING", {"default": "$list", "multiline": True}), 
                             "strings": ("STRING", {"multiline": True}), 
                             "idx": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                             }}
    RETURN_TYPES = ("STRING",)
    FUNCTION = "encode"

    CATEGORY = "conditioning"

    def encode(self, text, strings, idx):
        string_list = strings.split(",")
        wildcard = string_list[idx % len(string_list)].strip()
        return (text.replace("$list", wildcard), )
    

class Den_NoiseImage_AS:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "width": ("INT", {"default": 512, "min": 64, "max": MAX_RESOLUTION, "step": 64}),
                "height": ("INT", {"default": 512, "min": 64, "max": MAX_RESOLUTION, "step": 64}),
                "idx": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "doStuff"
    CATEGORY = "ASNodes"

    def doStuff(self, width, height, idx):
        new_image = torch.rand(
            (1, height, width, 3),
            dtype=torch.float32,
        )
        
        return (new_image,)

    


# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {
    "Den_ImageToLatentSpace": Den_ImageToLatentSpace,
    "Den_MaskToImage_AS": Den_MaskToImage_AS,
    "Den_ImageToMask_AS": Den_ImageToMask_AS,
    "Den_LatentMix_AS": Den_LatentMix_AS,
    "Den_LatentAdd_AS": Den_LatentAdd_AS,
    "Den_SaveLatent_AS": Den_SaveLatent_AS,
    "Den_LoadLatent_AS": Den_LoadLatent_AS,
    "Den_LatentToImages_AS": Den_LatentToImages_AS,
    "Den_LatentMixMasked_As": Den_LatentMixMasked_As,
    "Den_ImageMixMasked_As": Den_ImageMixMasked_As,
    "Den_TextToImage_AS": Den_TextToImage_AS,
    # "Den_BatchIndex_AS": Den_BatchIndex_AS,
    "Den_MapRange_AS": Den_MapRange_AS,
    "Den_Number_AS": Den_Number_AS,
    "Den_Int2Any_AS": Den_Int2Any_AS,
    "Den_Number2Int_AS": Den_Number2Int_AS,
    "Den_Number2Float_AS": Den_Number2Float_AS,
    "Den_Math_AS": Den_Math_AS,
    # "Increment_AS": Increment_AS,
    "Den_CropImage_AS": Den_CropImage_AS,
    "Den_TextWildcardList_AS": Den_TextWildcardList_AS,
    "Den_NoiseImage_AS": Den_NoiseImage_AS,
    "Den_Eval_AS": Den_Eval_AS,
}
