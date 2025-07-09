import gradio as gr
import argparse, torch, os
from PIL import Image
from src.tryon_pipeline import StableDiffusionXLInpaintPipeline as TryonPipeline
from src.unet_hacked_garmnet import UNet2DConditionModel as UNet2DConditionModel_ref
from src.unet_hacked_tryon import UNet2DConditionModel
from transformers import (
    CLIPImageProcessor,
    CLIPVisionModelWithProjection,
)
from diffusers import AutoencoderKL
from typing import List
from util.common import open_folder
from util.image import pil_to_binary_mask, save_output_image
from utils_mask import get_mask_location
from torchvision import transforms
import apply_net
from preprocess.humanparsing.run_parsing import Parsing
from preprocess.openpose.run_openpose import OpenPose
from detectron2.data.detection_utils import convert_PIL_to_numpy, _apply_exif_orientation
from torchvision.transforms.functional import to_pil_image
from util.pipeline import quantize_4bit, restart_cpu_offload, torch_gc
import sys
import replicate

# --- Cấu hình API Token từ biến môi trường (an toàn cho Kaggle/Github) ---
REPLICATE_API_TOKEN = os.getenv("REPLICATE_API_TOKEN")

if REPLICATE_API_TOKEN:
    os.environ["REPLICATE_API_TOKEN"] = REPLICATE_API_TOKEN
else:
    print("CẢNH BÁO: Không tìm thấy biến môi trường REPLICATE_API_TOKEN.")
    print("Chức năng tạo 3D sẽ bị vô hiệu hóa.")
    print("Vui lòng thêm API token của bạn vào Kaggle Secrets hoặc truyền trực tiếp vào lệnh chạy.")

# --- Các biến toàn cục và thiết lập ban đầu ---
ORIGINAL_IMAGE = None
parser = argparse.ArgumentParser()
parser.add_argument("--share", type=str, default=False, help="Set to True to share the app publicly.")
parser.add_argument("--lowvram", action="store_true", help="Enable CPU offload for model operations.")
parser.add_argument("--load_mode", default=None, type=str, choices=["4bit", "8bit"], help="Quantization mode")
parser.add_argument("--fixed_vae", action="store_true", default=True,  help="Use fixed vae for FP16.")
args = parser.parse_args()

load_mode = args.load_mode
dtype = torch.float16
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model_id = 'yisol/IDM-VTON'
dtypeQuantize = torch.float8_e4m3fn if load_mode in ('4bit','8bit') else dtype
ENABLE_CPU_OFFLOAD = args.lowvram
need_restart_cpu_offloading = False
unet, pipe, UNet_Encoder = None, None, None
example_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'example')


# --- Các hàm xử lý ---

def generate_3d_model(image_path: str):
    if not REPLICATE_API_TOKEN:
        gr.Warning("Không có Replicate API Token! Bỏ qua bước tạo 3D.")
        return None
    print(f"Bắt đầu gọi API Trellis cho ảnh: {image_path}")
    try:
        with open(image_path, "rb") as image_file:
            model_version = "firtoz/trellis:645e52126e55323539b71341d3d41575f05327b864a7465668e27c1010373413"
            output = replicate.run(model_version, input={"image": image_file})
            print(f"Tạo 3D thành công! URL: {output}")
            return output
    except Exception as e:
        print(f"Lỗi khi gọi API Replicate: {e}", file=sys.stderr)
        gr.Error(f"Lỗi khi gọi API Replicate. Chi tiết: {e}")
        return None

def auto_crop_upload(editor_value, crop_flag):
    global ORIGINAL_IMAGE
    if editor_value is None or editor_value.get("background") is None: return editor_value
    try:
        img = editor_value["background"].convert("RGB")
        if crop_flag:
            ORIGINAL_IMAGE = img.copy()
            width, height = img.size
            target_width, target_height = int(min(width, height * 3/4)), int(min(height, width * 4/3))
            left, top = (width - target_width) / 2, (height - target_height) / 2
            right, bottom = (width + target_width) / 2, (height + target_height) / 2
            resized_img = img.crop((left, top, right, bottom)).resize((768, 1024))
            editor_value["background"] = resized_img
            if editor_value.get("layers"):
                editor_value["layers"] = [l.crop((left, top, right, bottom)).resize((768, 1024)) if l else None for l in editor_value["layers"]]
            editor_value["composite"] = resized_img
            editor_value["auto_cropped"] = True
    except Exception as e:
        print(f"Lỗi auto crop: {e}", file=sys.stderr)
    return editor_value

def start_tryon(dict_img, garm_img, garment_des, category, is_checked, is_checked_crop, denoise_steps, is_randomize_seed, seed, number_of_images):
    global pipe, unet, UNet_Encoder, need_restart_cpu_offloading, ORIGINAL_IMAGE

    if pipe is None:
        unet = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet", torch_dtype=dtypeQuantize)
        if load_mode == '4bit': quantize_4bit(unet)
        unet.requires_grad_(False)
        image_encoder = CLIPVisionModelWithProjection.from_pretrained(model_id, subfolder="image_encoder", torch_dtype=torch.float16)
        if load_mode == '4bit': quantize_4bit(image_encoder)
        vae = AutoencoderKL.from_pretrained('madebyollin/sdxl-vae-fp16-fix', torch_dtype=dtype)
        UNet_Encoder = UNet2DConditionModel_ref.from_pretrained(model_id, subfolder="unet_encoder", torch_dtype=dtypeQuantize)
        if load_mode == '4bit': quantize_4bit(UNet_Encoder)
        UNet_Encoder.requires_grad_(False); image_encoder.requires_grad_(False); vae.requires_grad_(False); unet.requires_grad_(False)
        pipe_param = {'pretrained_model_name_or_path': model_id, 'unet': unet, 'torch_dtype': dtype, 'vae': vae, 'image_encoder': image_encoder, 'feature_extractor': CLIPImageProcessor()}
        pipe = TryonPipeline.from_pretrained(**pipe_param).to(device)
        pipe.unet_encoder = UNet_Encoder.to(pipe.unet.device)
        if load_mode == '4bit':
            if pipe.text_encoder: quantize_4bit(pipe.text_encoder)
            if pipe.text_encoder_2: quantize_4bit(pipe.text_encoder_2)
    elif ENABLE_CPU_OFFLOAD:
        need_restart_cpu_offloading = True

    torch_gc()
    parsing_model, openpose_model = Parsing(0), OpenPose(0)
    openpose_model.preprocessor.body_estimation.model.to(device)
    tensor_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
    if need_restart_cpu_offloading: restart_cpu_offload(pipe, load_mode)
    elif ENABLE_CPU_OFFLOAD: pipe.enable_model_cpu_offload()

    garm_img = garm_img.convert("RGB").resize((768, 1024))
    human_img_orig = dict_img["background"].convert("RGB")
    
    if is_checked_crop:
        orig = ORIGINAL_IMAGE if ORIGINAL_IMAGE is not None else human_img_orig
        orig_w, orig_h = orig.size; scale_factor = 1024 / orig_h
        final_background = orig.resize((int(orig_w * scale_factor), 1024))
        target_width, target_height = int(min(orig_w, orig_h * 3/4)), int(min(orig_h, orig_w * 4/3))
        left_orig, top_orig = (orig_w - target_width) / 2, (orig_h - target_height) / 2
        left_final, top_final = int(left_orig * scale_factor), int(top_orig * scale_factor)
        crop_size = (int(target_width * scale_factor), int(target_height * scale_factor))
        human_img = human_img_orig
    else:
        human_img = human_img_orig.resize((768, 1024))

    if is_checked:
        keypoints = openpose_model(human_img.resize((384, 512)))
        model_parse, _ = parsing_model(human_img.resize((384, 512)))
        mask, _ = get_mask_location('hd', category, model_parse, keypoints)
        mask = mask.resize((768, 1024))
    else:
        mask = pil_to_binary_mask(dict_img['layers'][0].resize((768, 1024))) if dict_img.get('layers') and dict_img['layers'][0] else Image.new('L', (768, 1024), 0)

    mask_gray = to_pil_image(((1 - transforms.ToTensor()(mask)) * tensor_transform(human_img) + 1.0) / 2.0)
    human_img_arg = convert_PIL_to_numpy(_apply_exif_orientation(human_img.resize((384, 512))), format="BGR")
    args_apply = apply_net.create_argument_parser().parse_args(('show', './configs/densepose_rcnn_R_50_FPN_s1x.yaml', './ckpt/densepose/model_final_162be9.pkl', 'dp_segm', '-v', '--opts', 'MODEL.DEVICE', 'cuda'))
    pose_img = Image.fromarray(args_apply.func(args_apply, human_img_arg)[:, :, ::-1]).resize((768, 1024))
    
    if pipe.text_encoder: pipe.text_encoder.to(device)
    if pipe.text_encoder_2: pipe.text_encoder_2.to(device)

    with torch.no_grad(), torch.cuda.amp.autocast(dtype=dtype):
        prompt_embeds, neg_embeds, pooled, neg_pooled = pipe.encode_prompt(f"model is wearing {garment_des}", num_images_per_prompt=1, do_classifier_free_guidance=True, negative_prompt="monochrome, lowres, bad anatomy, worst quality, low quality")
        prompt_embeds_c, _, _, _ = pipe.encode_prompt(f"a photo of {garment_des}", num_images_per_prompt=1, do_classifier_free_guidance=False)
        
        results, current_seed = [], seed
        for _ in range(number_of_images):
            if is_randomize_seed: current_seed = torch.randint(0, 2**32, (1,)).item()
            generator = torch.Generator(device).manual_seed(current_seed) if seed != -1 else None
            
            images = pipe(prompt_embeds=prompt_embeds, negative_prompt_embeds=neg_embeds, pooled_prompt_embeds=pooled, negative_pooled_prompt_embeds=neg_pooled, num_inference_steps=denoise_steps, generator=generator, strength=1.0, pose_img=tensor_transform(pose_img).unsqueeze(0).to(device, dtype), text_embeds_cloth=prompt_embeds_c, cloth=tensor_transform(garm_img).unsqueeze(0).to(device, dtype), mask_image=mask, image=human_img, height=1024, width=768, ip_adapter_image=garm_img, guidance_scale=2.0)[0]
            
            if is_checked_crop:
                final_img = final_background.copy()
                final_img.paste(images[0].resize(crop_size), (left_final, top_final))
                img_path = save_output_image(final_img, "outputs", 'img', current_seed)
            else:
                img_path = save_output_image(images[0], "outputs", 'img', current_seed)
            results.append(img_path)
            current_seed += 1
            
        model_3d_url = None
        if results:
            gr.Info("Đã tạo ảnh 2D. Bắt đầu quá trình tạo 3D (có thể mất vài phút)...")
            model_3d_url = generate_3d_model(results[0])
            if model_3d_url: gr.Info("Tạo mô hình 3D thành công!")
        
        return results, mask_gray, model_3d_url

# ====================================================================================================
# === SỬA LỖI: Di chuyển khối này lên trên để định nghĩa biến trước khi sử dụng ===
# --- Chuẩn bị danh sách ví dụ cho giao diện ---
garm_list_path = [os.path.join(example_path, "cloth", garm) for garm in os.listdir(os.path.join(example_path, "cloth"))]
human_list_paths = [os.path.join(example_path, "human", h) for h in os.listdir(os.path.join(example_path, "human"))]
human_ex_list = [{'background': p, 'layers': None, 'composite': None} for p in human_list_paths if "Jensen" in p or "sam1 (1)" in p]
# ====================================================================================================

# --- Xây dựng giao diện Gradio ---
with gr.Blocks().queue() as demo:
    gr.Markdown("## IDM-VTON with 2D-to-3D Conversion")
    gr.Markdown("Virtual Try-on with your image and garment. The 2D result will be converted to a 3D model using Replicate's Trellis API.")
    
    with gr.Row():
        with gr.Column(scale=1):
            imgs = gr.ImageEditor(sources='upload', type="pil", label='Human Image', interactive=True, height=550)
            garm_img = gr.Image(label="Garment", sources='upload', type="pil", height=550)
        
        with gr.Column(scale=1):
            with gr.Accordion("Settings & Examples", open=True):
                prompt = gr.Textbox(placeholder="Description of garment, e.g., 'red short sleeve t-shirt'", label="Garment Description")
                category = gr.Radio(["upper_body", "lower_body", "dresses"], label="Garment Category", value="upper_body")
                with gr.Row():
                    is_checked = gr.Checkbox(label="Auto-masking", value=True)
                    is_checked_crop = gr.Checkbox(label="Auto-crop", value=True)
                gr.Examples(inputs=imgs, examples_per_page=4, examples=human_ex_list, label="Human Examples")
                gr.Examples(inputs=garm_img, examples_per_page=8, examples=garm_list_path, label="Garment Examples")
            
            with gr.Accordion("Advanced Settings", open=False):
                denoise_steps = gr.Number(label="Denoising Steps", minimum=20, maximum=120, value=30, step=1)
                seed = gr.Number(label="Seed", minimum=-1, maximum=2147483647, step=1, value=1)
                is_randomize_seed = gr.Checkbox(label="Randomize seed", value=True)
                number_of_images = gr.Number(label="Number Of Images", minimum=1, maximum=10, value=1, step=1)
            
            try_button = gr.Button("Start Try-on & Generate 3D", variant="primary")
            btn_open_outputs = gr.Button("Open Outputs Folder")
        
        with gr.Column(scale=2):
            with gr.Tabs():
                with gr.TabItem("2D & Mask Output"):
                    image_gallery = gr.Gallery(label="Generated Images", show_label=True, columns=2, height='auto')
                    masked_img = gr.Image(label="Masked Image", show_share_button=False)
                with gr.TabItem("3D Output (Trellis)"):
                    model_3d_output = gr.Model3D(label="3D Model from first image", interactive=False, height=700)
    
    # --- Kết nối các sự kiện ---
    imgs.upload(auto_crop_upload, inputs=[imgs, is_checked_crop], outputs=imgs)
    btn_open_outputs.click(fn=open_folder)
    try_button.click(fn=start_tryon, 
                     inputs=[imgs, garm_img, prompt, category, is_checked, is_checked_crop, denoise_steps, is_randomize_seed, seed, number_of_images], 
                     outputs=[image_gallery, masked_img, model_3d_output], 
                     api_name='tryon')

demo.launch(inbrowser=True, share=args.share)
