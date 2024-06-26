import torch
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import sys, os
# facerestore
from comfy_extras.chainner_models import model_loading
from comfy import model_management
from custom_nodes.facerestore_cf.basicsr.utils.registry import ARCH_REGISTRY
import cv2
from custom_nodes.facerestore_cf.facelib.utils.face_restoration_helper import FaceRestoreHelper
from torchvision.transforms.functional import normalize
import math
# Latent
import comfy.model_management  # CUDA detect
# LLM
from llama_cpp import Llama
from llama_cpp.llama_chat_format import Llava15ChatHandler
import folder_paths
from io import BytesIO
import base64

MAX_RESOLUTION = 8192

# LLM Special Load
models_base_path = os.path.join(folder_paths.models_dir, "GPTcheckpoints")
_folders_whitelist = ["moondream", "joytag"]  # ,"internlm"]


def get_model_list(models_base_path, supported_gpt_extensions):
    all_models = []
    try:
        for file in os.listdir(models_base_path):

            if os.path.isdir(os.path.join(models_base_path, file)):
                if file in _folders_whitelist:
                    all_models.append(os.path.join(models_base_path, file))

            else:
                if file.endswith(tuple(supported_gpt_extensions)):
                    all_models.append(os.path.join(models_base_path, file))
    except:
        print(f"Path {models_base_path} not valid.")
    return all_models


def get_model_path(folder_list, model_name):
    for folder_path in folder_list:
        if folder_path.endswith(model_name):
            return folder_path


supported_gpt_extensions = set(['.gguf'])
supported_clip_extensions = set(['.gguf', '.bin'])
model_external_path = None
all_models = []
try:
    model_external_path = folder_paths.folder_names_and_paths["GPTcheckpoints"][0][0]
except:
    # no external folder
    pass

all_llava_models = get_model_list(os.path.join(folder_paths.models_dir, "GPTcheckpoints", "llava", "models"),
                                  supported_gpt_extensions)
all_llava_clips = get_model_list(os.path.join(folder_paths.models_dir, "GPTcheckpoints", "llava", "clips"),
                                 supported_clip_extensions)

all_models = get_model_list(models_base_path, supported_gpt_extensions)
if model_external_path is not None:
    all_models += get_model_list(model_external_path, supported_gpt_extensions)
all_models += all_llava_models
# extract only names
all_models_names = [os.path.basename(model) for model in all_models]

all_clips_names = [os.path.basename(model) for model in all_llava_clips]


class Den_GPTLoaderSimple_llama:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
            {
                "ckpt_name": (all_models_names,),
                "clip_name": (all_clips_names,),
                "gpu_layers": ("INT", {"default": 27, "min": 0, "max": 100, "step": 1}),
                "n_threads": ("INT", {"default": 8, "min": 1, "max": 100, "step": 1}),
                "max_ctx": ("INT", {"default": 2048, "min": 300, "max": 100000, "step": 4}),
            },
        }

    RETURN_TYPES = ("CUSTOM",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load_gpt_checkpoint"

    CATEGORY = "LLM"

    def load_gpt_checkpoint(self, ckpt_name, clip_name, gpu_layers, n_threads, max_ctx, llava_clip=None):
        ckpt_path = get_model_path(all_models, ckpt_name)
        llm = None
        clip_path = get_model_path(all_llava_clips, clip_name)
        llava_clip = Llava15ChatHandler(clip_model_path=clip_path, verbose=False)
        # if is path
        if os.path.isfile(ckpt_path):
            print("GPT MODEL DETECTED")
            if "llava" in ckpt_path:
                if llava_clip is None:
                    raise ValueError("Please provide a llava clip")
                llm = Llama(model_path=ckpt_path, n_gpu_layers=gpu_layers, verbose=False, n_threads=n_threads,
                            n_ctx=max_ctx, logits_all=True, chat_handler=llava_clip)
            else:
                llm = Llama(model_path=ckpt_path, n_gpu_layers=gpu_layers, verbose=False, n_threads=n_threads,
                            n_ctx=max_ctx)
        return ([llm, ckpt_name, ckpt_path],)


class Den_GPTSampler_llama:
    """
    A custom node by Den for text generation using GPT

    Attributes
    ----------
    max_tokens (`int`): Maximum number of tokens in the generated text.
    temperature (`float`): Temperature parameter for controlling randomness (0.2 to 1.0).
    top_p (`float`): Top-p probability for nucleus sampling.
    logprobs (`int`|`None`): Number of log probabilities to output alongside the generated text.
    echo (`bool`): Whether to print the input prompt alongside the generated text.
    stop (`str`|`List[str]`|`None`): Tokens at which to stop generation.
    frequency_penalty (`float`): Frequency penalty for word repetition.
    presence_penalty (`float`): Presence penalty for word diversity.
    repeat_penalty (`float`): Penalty for repeating a prompt's output.
    top_k (`int`): Top-k tokens to consider during generation.
    stream (`bool`): Whether to generate the text in a streaming fashion.
    tfs_z (`float`): Temperature scaling factor for top frequent samples.
    model (`str`): The GPT model to use for text generation.
    """

    def __init__(self):
        self.temp_prompt = ""
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {

                "model": ("CUSTOM", {"default": ""}),
                "max_tokens": ("INT", {"default": 2048}),
                "temperature": ("FLOAT", {"default": 0.7, "min": 0.2, "max": 1.0}),
                "top_p": ("FLOAT", {"default": 0.5, "min": 0.1, "max": 1.0}),
                "logprobs": ("INT", {"default": 0}),
                "echo": (["enable", "disable"], {"default": "disable"}),
                "stop_token": ("STRING", {"default": "STOPTOKEN"}),
                "frequency_penalty": ("FLOAT", {"default": 0.0}),
                "presence_penalty": ("FLOAT", {"default": 0.0}),
                "repeat_penalty": ("FLOAT", {"default": 1.17647}),
                "top_k": ("INT", {"default": 40}),
                "tfs_z": ("FLOAT", {"default": 1.0}),
                "print_output": (["enable", "disable"], {"default": "disable"}),
                "cached": (["YES", "NO"], {"default": "NO"}),
                "prefix": ("STRING", {"default": "### Instruction: "}),
                "suffix": ("STRING", {"default": "### Response: "}),
                "max_tags": ("INT", {"default": 50}),

            },
            "optional": {
                "prompt": ("STRING", {"forceInput": True}),
                "image": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("STRING",)
    OUTPUT_IS_LIST = (True,)
    FUNCTION = "generate_text"
    CATEGORY = "LLM"

    def generate_text(self, max_tokens, temperature, top_p, logprobs, echo, stop_token, frequency_penalty,
                      presence_penalty, repeat_penalty, top_k, tfs_z, model, print_output, cached, prefix, suffix,
                      max_tags, image=None, prompt=None):
        model_funct = model[0]
        model_name = model[1]
        model_path = model[2]

        if cached == "NO":
            if "llava" in model_path:
                print(f"Den Used Llama Model!: {prompt}\n")
                cont = llava_inference(model_funct, prompt, image, max_tokens, stop_token, frequency_penalty,
                                       presence_penalty, repeat_penalty, temperature, top_k, top_p)
                self.temp_prompt = cont
            else:
                print(f"Den Used Wrong!: {prompt}\n")
                # Call your GPT generation function here using the provided parameters
                composed_prompt = f"{prefix} {prompt} {suffix}"
                cont = ""
                stream = model_funct(max_tokens=max_tokens, stop=[stop_token], stream=False,
                                     frequency_penalty=frequency_penalty, presence_penalty=presence_penalty,
                                     repeat_penalty=repeat_penalty, temperature=temperature, top_k=top_k,
                                     top_p=top_p, model=model_path, prompt=composed_prompt)
                cont = [stream["choices"][0]["text"]]
                self.temp_prompt = cont
        else:
            cont = self.temp_prompt
            # remove fist 30 characters of cont
        try:
            if print_output == "enable":
                print(f"Input: {prompt}\nGenerated Text: {cont}")
            return {"ui": {"text": cont}, "result": (cont,)}

        except:
            if print_output == "enable":
                print(f"Input: {prompt}\nGenerated Text: ")
            return {"ui": {"text": " "}, "result": (" ",)}


def tensor2pil(image):
    return Image.fromarray(np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))


# Convert PIL to Tensor
def pil2tensor(image):
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)


def detect_device():
    """
    Detects the appropriate device to run on, and return the device and dtype.
    """
    if torch.cuda.is_available():
        return torch.device("cuda"), torch.float16
    elif torch.backends.mps.is_available():
        return torch.device("mps"), torch.float16
    else:
        return torch.device("cpu"), torch.float32


def llava_inference(model_funct, prompt, images, max_tokens, stop_token, frequency_penalty, presence_penalty,
                    repeat_penalty, temperature, top_k, top_p):
    list_descriptions = []
    for image in images:
        pil_image = tensor2pil(image)
        # Convert the PIL image to a bytes buffer
        buffer = BytesIO()
        pil_image.save(buffer, format="JPEG")  # You can change the format if needed
        image_bytes = buffer.getvalue()
        base64_string = f"data:image/jpeg;base64,{base64.b64encode(image_bytes).decode('utf-8')}"

        response = model_funct.create_chat_completion(max_tokens=max_tokens, stop=[stop_token], stream=False,
                                                      frequency_penalty=frequency_penalty,
                                                      presence_penalty=presence_penalty,
                                                      repeat_penalty=repeat_penalty,
                                                      temperature=temperature, top_k=top_k, top_p=top_p,
                                                      messages=[
                                                          {"role": "system",
                                                           "content": "You are an assistant who perfectly describes images."},
                                                          {
                                                              "role": "user",
                                                              "content": [
                                                                  {"type": "image_url",
                                                                   "image_url": {"url": base64_string}},
                                                                  {"type": "text", "text": prompt}
                                                              ]
                                                          }
                                                      ]
                                                      )
        list_descriptions.append(response['choices'][0]['message']['content'])
    return list_descriptions


# SVD from size Image
class Den_SVD_img2vid:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"clip_vision": ("CLIP_VISION",),
                             "init_image": ("IMAGE",),
                             "vae": ("VAE",),
                             "video_frames": ("INT", {"default": 14, "min": 1, "max": 4096}),
                             "motion_bucket_id": ("INT", {"default": 127, "min": 1, "max": 1023}),
                             "fps": ("INT", {"default": 6, "min": 1, "max": 1024}),
                             "augmentation_level": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 10.0, "step": 0.01})
                             }}

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "LATENT")
    RETURN_NAMES = ("positive", "negative", "latent")

    FUNCTION = "encode"

    CATEGORY = "conditioning/video_models"

    def encode(self, clip_vision, init_image, vae, video_frames, motion_bucket_id, fps, augmentation_level):
        height = init_image.shape[1]
        width = init_image.shape[2]
        output = clip_vision.encode_image(init_image)
        pooled = output.image_embeds.unsqueeze(0)
        pixels = comfy.utils.common_upscale(init_image.movedim(-1, 1), width, height, "bilinear", "center").movedim(1,
                                                                                                                    -1)
        encode_pixels = pixels[:, :, :, :3]
        if augmentation_level > 0:
            encode_pixels += torch.randn_like(pixels) * augmentation_level
        t = vae.encode(encode_pixels)
        positive = [[pooled,
                     {"motion_bucket_id": motion_bucket_id, "fps": fps, "augmentation_level": augmentation_level,
                      "concat_latent_image": t}]]
        negative = [[torch.zeros_like(pooled),
                     {"motion_bucket_id": motion_bucket_id, "fps": fps, "augmentation_level": augmentation_level,
                      "concat_latent_image": torch.zeros_like(t)}]]
        latent = torch.zeros([video_frames, 4, height // 8, width // 8])
        return (positive, negative, {"samples": latent})


# Latent space from size Image
class Den_ImageToLatentSpace:
    @classmethod
    def INPUT_TYPES(s):
        return \
            {"required":
                {
                    "image": ("IMAGE",),
                    "batch_size": ("INT", {"default": 1, "min": 1, "max": 4096}),
                    "upscale_factor": ("FLOAT", {"default": 1.0, "min": 0.01, "max": 8.0, "step": 0.1})
                }
            }

    RETURN_TYPES = ("LATENT", "image")
    RETURN_TYPES = ("LATENT", "IMAGE")
    FUNCTION = "generate"

    CATEGORY = "latent"

    def generate(self, image, batch_size=1, upscale_factor=1):
        upscale_methods = ["nearest-exact", "bilinear", "area", "bicubic", "lanczos"]
        new_image = image.movedim(-1, 1)
        width = round(new_image.shape[3] * upscale_factor)
        height = round(new_image.shape[2] * upscale_factor)
        image2 = comfy.utils.common_upscale(new_image, width, height, upscale_methods[0], "disabled")
        image = image2.movedim(1, -1)
        height = image.shape[1]
        width = image.shape[2]
        latent = torch.zeros([batch_size, 4, height // 8, width // 8],
                             device=comfy.model_management.intermediate_device())
        return ({"samples": latent}, image,)


def img2tensor(imgs, bgr2rgb=True, float32=True):
    """Numpy array to tensor.

    Args:
        imgs (list[ndarray] | ndarray): Input images.
        bgr2rgb (bool): Whether to change bgr to rgb.
        float32 (bool): Whether to change to float32.

    Returns:
        list[tensor] | tensor: Tensor images. If returned results only have
            one element, just return tensor.
    """

    def _totensor(img, bgr2rgb, float32):
        if img.shape[2] == 3 and bgr2rgb:
            if img.dtype == 'float64':
                img = img.astype('float32')
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = torch.from_numpy(img.transpose(2, 0, 1))
        if float32:
            img = img.float()
        return img

    if isinstance(imgs, list):
        return [_totensor(img, bgr2rgb, float32) for img in imgs]
    else:
        return _totensor(imgs, bgr2rgb, float32)


def tensor2img(tensor, rgb2bgr=True, out_type=np.uint8, min_max=(0, 1)):
    """Convert torch Tensors into image numpy arrays.

    After clamping to [min, max], values will be normalized to [0, 1].

    Args:
        tensor (Tensor or list[Tensor]): Accept shapes:
            1) 4D mini-batch Tensor of shape (B x 3/1 x H x W);
            2) 3D Tensor of shape (3/1 x H x W);
            3) 2D Tensor of shape (H x W).
            Tensor channel should be in RGB order.
        rgb2bgr (bool): Whether to change rgb to bgr.
        out_type (numpy type): output types. If ``np.uint8``, transform outputs
            to uint8 type with range [0, 255]; otherwise, float type with
            range [0, 1]. Default: ``np.uint8``.
        min_max (tuple[int]): min and max values for clamp.

    Returns:
        (Tensor or list): 3D ndarray of shape (H x W x C) OR 2D ndarray of
        shape (H x W). The channel order is BGR.
    """
    if not (torch.is_tensor(tensor) or (isinstance(tensor, list) and all(torch.is_tensor(t) for t in tensor))):
        raise TypeError(f'tensor or list of tensors expected, got {type(tensor)}')

    if torch.is_tensor(tensor):
        tensor = [tensor]
    result = []
    for _tensor in tensor:
        _tensor = _tensor.squeeze(0).float().detach().cpu().clamp_(*min_max)
        _tensor = (_tensor - min_max[0]) / (min_max[1] - min_max[0])

        n_dim = _tensor.dim()
        if n_dim == 4:
            img_np = make_grid(_tensor, nrow=int(math.sqrt(_tensor.size(0))), normalize=False).numpy()
            img_np = img_np.transpose(1, 2, 0)
            if rgb2bgr:
                img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        elif n_dim == 3:
            img_np = _tensor.numpy()
            img_np = img_np.transpose(1, 2, 0)
            if img_np.shape[2] == 1:  # gray image
                img_np = np.squeeze(img_np, axis=2)
            else:
                if rgb2bgr:
                    img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        elif n_dim == 2:
            img_np = _tensor.numpy()
        else:
            raise TypeError('Only support 4D, 3D or 2D tensor. ' f'But received with dimension: {n_dim}')
        if out_type == np.uint8:
            # Unlike MATLAB, numpy.unit8() WILL NOT round by default.
            img_np = (img_np * 255.0).round()
        img_np = img_np.astype(out_type)
        result.append(img_np)
    if len(result) == 1:
        result = result[0]
    return result


class Den_FaceRestoreCFWithModel:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "model_name": (folder_paths.get_filename_list("facerestore_models"),),
            "image": ("IMAGE",),
            "facedetection": (["retinaface_resnet50", "retinaface_mobile0.25", "YOLOv5l", "YOLOv5n"],),
            "codeformer_fidelity": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1, "step": 0.05})
        }}

    RETURN_TYPES = ("IMAGE",)

    FUNCTION = "restore_face"

    CATEGORY = "facerestore_cf"

    def __init__(self):
        self.face_helper = None

    def restore_face(self, model_name, image, facedetection, codeformer_fidelity):
        if "codeformer" in model_name.lower():
            print(f'\tLoading CodeFormer: {model_name}')
            model_path = folder_paths.get_full_path("facerestore_models", model_name)
            device = model_management.get_torch_device()
            codeformer_net = ARCH_REGISTRY.get("CodeFormer")(
                dim_embd=512,
                codebook_size=1024,
                n_head=8,
                n_layers=9,
                connect_list=["32", "64", "128", "256"],
            ).to(device)
            checkpoint = torch.load(model_path)["params_ema"]
            codeformer_net.load_state_dict(checkpoint)
            out = codeformer_net.eval()
        else:
            model_path = folder_paths.get_full_path("facerestore_models", model_name)
            sd = comfy.utils.load_torch_file(model_path, safe_load=True)
            out = model_loading.load_state_dict(sd).eval()
        facerestore_model = out
        print(f'\tStarting restore_face with codeformer_fidelity: {codeformer_fidelity}')
        device = model_management.get_torch_device()
        facerestore_model.to(device)
        if self.face_helper is None:
            self.face_helper = FaceRestoreHelper(1, face_size=512, crop_ratio=(1, 1), det_model=facedetection,
                                                 save_ext='png', use_parse=True, device=device)

        image_np = 255. * image.cpu().numpy()

        total_images = image_np.shape[0]
        out_images = np.ndarray(shape=image_np.shape)

        for i in range(total_images):
            cur_image_np = image_np[i, :, :, ::-1]

            original_resolution = cur_image_np.shape[0:2]

            if facerestore_model is None or self.face_helper is None:
                return image

            self.face_helper.clean_all()
            self.face_helper.read_image(cur_image_np)
            self.face_helper.get_face_landmarks_5(only_center_face=False, resize=640, eye_dist_threshold=5)
            self.face_helper.align_warp_face()

            restored_face = None
            for idx, cropped_face in enumerate(self.face_helper.cropped_faces):
                cropped_face_t = img2tensor(cropped_face / 255., bgr2rgb=True, float32=True)
                normalize(cropped_face_t, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)
                cropped_face_t = cropped_face_t.unsqueeze(0).to(device)

                try:
                    with torch.no_grad():
                        # output = facerestore_model(cropped_face_t, w=strength, adain=True)[0]
                        # output = facerestore_model(cropped_face_t)[0]
                        output = facerestore_model(cropped_face_t, w=codeformer_fidelity)[0]
                        restored_face = tensor2img(output, rgb2bgr=True, min_max=(-1, 1))
                    del output
                    torch.cuda.empty_cache()
                except Exception as error:
                    print(f'\tFailed inference for CodeFormer: {error}', file=sys.stderr)
                    restored_face = tensor2img(cropped_face_t, rgb2bgr=True, min_max=(-1, 1))

                restored_face = restored_face.astype('uint8')
                self.face_helper.add_restored_face(restored_face)

            self.face_helper.get_inverse_affine(None)

            restored_img = self.face_helper.paste_faces_to_input_image()
            restored_img = restored_img[:, :, ::-1]

            if original_resolution != restored_img.shape[0:2]:
                restored_img = cv2.resize(restored_img, (0, 0), fx=original_resolution[1] / restored_img.shape[1],
                                          fy=original_resolution[0] / restored_img.shape[0],
                                          interpolation=cv2.INTER_LINEAR)

            self.face_helper.clean_all()

            # restored_img = cv2.cvtColor(restored_face, cv2.COLOR_BGR2RGB)

            out_images[i] = restored_img

        restored_img_np = np.array(out_images).astype(np.float32) / 255.0
        restored_img_tensor = torch.from_numpy(restored_img_np)
        return (restored_img_tensor,)


class Den_MaskToImage_AS:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"mask": ("MASK",), }}

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
        return {"required": {"image": ("IMAGE",), }}

    RETURN_TYPES = ("MASK",)
    FUNCTION = "convert"
    CATEGORY = "ASNodes"

    def convert(self, image):
        return (image.squeeze().mean(2),)


class Den_LatentMix_AS:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"samples_to": ("LATENT",),
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
        return {"required": {"samples_to": ("LATENT",),
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
        return {"required": {"latent_in": ("LATENT",), }}

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "doStuff"
    CATEGORY = "ASNodes"

    def doStuff(self, latent_in):
        torch.save(latent_in, 'latent.pt')
        return (latent_in,)


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

    def doStuff(self, ):
        latent_out = torch.load('latent.pt')
        return (latent_out,)


class Den_LatentToImages_AS:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"latent_in": ("LATENT",), }}

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
                images_out[idx, :, :, chan] = s[0, idx, :, :]
        return (images_out,)


class Den_LatentMixMasked_As:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"samples_to": ("LATENT",),
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
        return {"required": {"image_to": ("IMAGE",),
                             "image_from": ("IMAGE",),
                             "mask": ("MASK",),
                             }}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "composite"
    CATEGORY = "ASNodes"

    def composite(self, image_to, image_from, mask):
        image_out = image_to.clone()
        image_out[0, :, :, 0] = image_to[0, :, :, 0] * mask + image_from[0, :, :, 0] * (1 - mask)
        image_out[0, :, :, 1] = image_to[0, :, :, 1] * mask + image_from[0, :, :, 1] * (1 - mask)
        image_out[0, :, :, 2] = image_to[0, :, :, 2] * mask + image_from[0, :, :, 2] * (1 - mask)
        return (image_out,)


class Den_TextToImage_AS:
    @classmethod
    def INPUT_TYPES(s):
        fonts = os.listdir('C:\Windows\Fonts')

        return {
            "required": {
                "text": ("STRING", {"multiline": True}),
                "font": (fonts,),
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
        draw.text((x, y), text, font=font, fill=(255, 255, 255))

        np_image = np.array(PIL_image)

        new_image = torch.zeros(
            (1, height, width, 3),
            dtype=torch.float32,
        )
        new_image[0, :, :, :] = torch.from_numpy(np_image) / 256

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
                "value": ("number", {}),
            },
        }

    RETURN_TYPES = ("FLOAT",)
    FUNCTION = "convert"
    CATEGORY = "ASNodes"

    def convert(self, value):
        return (value,)


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
        return (value,)


class Den_Number2Int_AS:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "value": ("number", {}),
            },
        }

    RETURN_TYPES = ("INT",)
    FUNCTION = "convert"
    CATEGORY = "ASNodes"

    def convert(self, value):
        return (round(value),)


class Den_Eval_AS:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "optional": {
                "i1": ("INT", {"default": 0, "step": 1}),
                "i2": ("INT", {"default": 0, "step": 1}),
                "f1": ("FLOAT", {"default": 0.0, "step": 0.1, "round": 0.01}),
                "f2": ("FLOAT", {"default": 0.0, "step": 0.1, "round": 0.01}),
                "s1": ("STRING", {"multiline": True, "default": ""}),
                "s2": ("STRING", {"multiline": True, "default": ""}),
            },
            "required": {
                "int_prc": ("STRING", {"multiline": False, "default": "0"}),
                "float_prc": ("STRING", {"multiline": False, "default": "0"}),
                "str_prc": ("STRING", {"multiline": False, "default": "0"}),
            },

        }

    RETURN_TYPES = ("INT", "FLOAT", "STRING",)
    FUNCTION = "do_stuff"
    CATEGORY = "ASNodes"

    def do_stuff(self, int_prc, float_prc, str_prc, i1, i2, f1, f2, s1, s2):
        int_out = eval(int_prc)
        float_out = eval(float_prc)
        str_out = eval(str_prc)

        return (int_out, float_out, str_out)


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
        return (value,)


class Den_Math_AS:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "do": (["add", "subtract", "multiply", "divide", "power"],),
                "in_0": ("FLOAT", {"default": 0, "step": 0.01, "round": 0.01, }),
                "in_1": ("FLOAT", {"default": 0, "step": 0.01, "round": 0.01, }),
            },
        }

    RETURN_TYPES = ("FLOAT", "INT")
    FUNCTION = "calculate"
    CATEGORY = "ASNodes"

    def calculate(self, do, in_0, in_1):
        if do == "add":
            result = in_0 + in_1
        if do == "subtract":
            result = in_0 - in_1
        if do == "multiply":
            result = in_0 * in_1
        if do == "divide":
            result = in_0 / in_1
        if do == "power":
            result = in_0 ** in_1

        return (result, round(result))


class Den_Increment_AS:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "value": ("INT", {"default": 10, "min": 0, "max": 20, "step": 3}),
            },
        }

    RETURN_TYPES = ("number",)
    FUNCTION = "doStuff"
    CATEGORY = "ASNodes"

    def doStuff(self, value):
        return (value,)


class Den_CropImage_AS:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"image": ("IMAGE",),
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
        image_out = image_out[:, y:y + height, x:x + width]
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
        return (text.replace("$list", wildcard),)


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
    "Den_GPTLoaderSimple_llama": Den_GPTLoaderSimple_llama,
    "Den_GPTSampler_llama": Den_GPTSampler_llama,
    "Den_SVD_img2vid": Den_SVD_img2vid,
    "Den_ImageToLatentSpace": Den_ImageToLatentSpace,
    "Den_FaceRestoreCFWithModel": Den_FaceRestoreCFWithModel,
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
