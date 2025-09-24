import os
import argparse
import numpy as np
import torch
import torch.optim as optim
from PIL import Image
from skimage.io import imsave
from utils import *
import argparse
import os
import piqa
import copy

from torchvision.ops import box_convert

# Grounding DINO
# import GroundingDINO.groundingdino.datasets.transforms as T
# from GroundingDINO.groundingdino.models import build_model
# from GroundingDINO.groundingdino.util import box_ops
# from GroundingDINO.groundingdino.util.slconfig import SLConfig
# from GroundingDINO.groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap
# from GroundingDINO.groundingdino.util.inference import annotate, load_image, predict


import groundingdino.datasets.transforms as T
from groundingdino.models import build_model
from groundingdino.util import box_ops
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap
from groundingdino.util.inference import annotate, load_image, predict

import supervision as sv

# segment anything
from segment_anything import build_sam, SamPredictor
import cv2
import numpy as np
import matplotlib.pyplot as plt


# diffusers
import PIL
import requests
import torch
from io import BytesIO
import time
import warnings
warnings.filterwarnings("ignore", category=UserWarning)



from huggingface_hub import hf_hub_download

# Argument parser
parser = argparse.ArgumentParser()
parser.add_argument('--source_file', type=str, default='data/1_source.png', help='Path to the source image')
parser.add_argument('--mask_file', type=str, default='data/1_mask.png', help='Path to the mask image')
parser.add_argument('--target_file', type=str, default='data/1_target.png', help='Path to the target image')
parser.add_argument('--output_dir', type=str, default='results/demo', help='Output directory')
parser.add_argument('--background_dir', type=str, default='./', help='Output directory')
parser.add_argument('--ss', type=int, default=512, help='Source image size')
parser.add_argument('--ts', type=int, default=512, help='Target image size')
parser.add_argument('--x', type=int, default=256, help='Vertical location (center)')
parser.add_argument('--y', type=int, default=256, help='Horizontal location (center)')
parser.add_argument('--gpu_id', type=int, default=0, help='GPU ID')
parser.add_argument('--num_steps', type=int, default=1000, help='Number of iterations')
parser.add_argument('--save_video', type=bool, default=False, help='Save the intermediate reconstruction process')
parser.add_argument('--background', type=str, default='data/1_background.png', help='Background image')
parser.add_argument('--new_id', type=str, default='data/1_new_id.png', help='New ID image')
opt = parser.parse_args()

ckpt_repo_id = "ShilongLiu/GroundingDINO"
ckpt_filenmae = "groundingdino_swinb_cogcoor.pth"
ckpt_config_filename = "GroundingDINO_SwinB.cfg.py"

def load_model_hf(repo_id, filename, ckpt_config_filename, device='cpu'):
	cache_config_file = hf_hub_download(repo_id=repo_id, filename=ckpt_config_filename)

	args = SLConfig.fromfile(cache_config_file)
	model = build_model(args)
	args.device = device

	cache_file = hf_hub_download(repo_id=repo_id, filename=filename)
	checkpoint = torch.load(cache_file, map_location='cpu')
	log = model.load_state_dict(clean_state_dict(checkpoint['model']), strict=False)
	print("Model loaded from {} \n => {}".format(cache_file, log))
	_ = model.eval()
	return model

groundingdino_model = load_model_hf(ckpt_repo_id, ckpt_filenmae, ckpt_config_filename)

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

sam_checkpoint = 'sam_vit_h_4b8939.pth'
sam = build_sam(checkpoint=sam_checkpoint)
sam.to(device=DEVICE)
sam_predictor = SamPredictor(sam)

psnr = piqa.PSNR().cuda()
ssim = piqa.SSIM().cuda()

def run(transformed_id_image, background_img, mask_image):
	# Create output directory
	os.makedirs(opt.output_dir, exist_ok=True)

	gpu_id = opt.gpu_id
	num_steps = opt.num_steps
	ss = opt.ss; # source image size
	ts = opt.ts # target image size
	x_start = opt.x; y_start = opt.y # blending location

	# Default weights for loss functions in the first pass
	grad_weight = 1e3; style_weight = 1e4; content_weight = 2; tv_weight = 1e-2; hist_weight = 5e-1; psnr_weight = 1e6; ssim_weight = 1e6

	# Set device
	device = torch.device(f"cuda:{opt.gpu_id}" if torch.cuda.is_available() else "cpu")
	torch.cuda.set_device(opt.gpu_id)

	source_img = np.array(Image.open(transformed_id_image).convert('RGB').resize((ss, ss)))
	target_img = np.array(Image.open(background_img).convert('RGB').resize((ts, ts)))
	mask_img = np.array(Image.open(mask_image).convert('L').resize((ss, ss)))
	mask_img[mask_img>0] = 1

	# Make Canvas Mask
	canvas_mask = make_canvas_mask(x_start, y_start, target_img, mask_img)
	canvas_mask = numpy2tensor(canvas_mask, gpu_id)
	canvas_mask = canvas_mask.squeeze(0).repeat(3,1).view(3,ts,ts).unsqueeze(0)

	# Compute Ground-Truth Gradients
	gt_gradient = compute_gt_gradient(x_start, y_start, source_img, target_img, mask_img, gpu_id)

	# Convert Numpy Images Into Tensors
	source_img = torch.from_numpy(source_img).unsqueeze(0).transpose(1,3).transpose(2,3).float().to(gpu_id)
	target_img = torch.from_numpy(target_img).unsqueeze(0).transpose(1,3).transpose(2,3).float().to(gpu_id)
	input_img = torch.randn(target_img.shape).to(gpu_id)

	mask_img = numpy2tensor(mask_img, gpu_id)
	mask_img = mask_img.squeeze(0).repeat(3,1).view(3,ss,ss).unsqueeze(0)

	# Define LBFGS optimizer
	def get_input_optimizer(input_img):
		optimizer = optim.LBFGS([input_img.requires_grad_()])
		return optimizer
	optimizer = get_input_optimizer(input_img)

	# Define Loss Functions
	mse = torch.nn.MSELoss()

	# Import VGG network for computing style and content loss
	mean_shift = MeanShift(gpu_id)
	vgg = Vgg16().to(gpu_id)

	run = [0]
	while run[0] <= num_steps:
		
		def closure():
			# Composite Foreground and Background to Make Blended Image
			blend_img = torch.zeros(target_img.shape).to(gpu_id)
			blend_img = input_img*canvas_mask + target_img*(canvas_mask-1)*(-1) 
			
			# Compute Laplacian Gradient of Blended Image
			pred_gradient = laplacian_filter_tensor(blend_img, gpu_id)
			
			# Compute Gradient Loss
			grad_loss = 0
			for c in range(len(pred_gradient)):
				grad_loss += mse(pred_gradient[c], gt_gradient[c])
			grad_loss /= len(pred_gradient)
			grad_loss *= grad_weight
			
			# Compute Style Loss
			target_features_style = vgg(mean_shift(target_img))
			target_gram_style = [gram_matrix(y) for y in target_features_style]
			
			blend_features_style = vgg(mean_shift(input_img))
			blend_gram_style = [gram_matrix(y) for y in blend_features_style]
			
			style_loss = 0
			for layer in range(len(blend_gram_style)):
				style_loss += mse(blend_gram_style[layer], target_gram_style[layer])
			style_loss /= len(blend_gram_style)  
			style_loss *= style_weight           

			
			# Compute Content Loss
			blend_obj = blend_img[:,:,int(x_start-source_img.shape[2]*0.5):int(x_start+source_img.shape[2]*0.5), int(y_start-source_img.shape[3]*0.5):int(y_start+source_img.shape[3]*0.5)]
			source_object_features = vgg(mean_shift(source_img*mask_img))
			blend_object_features = vgg(mean_shift(blend_obj*mask_img))
			content_loss = content_weight * mse(blend_object_features.relu2_2, source_object_features.relu2_2)
			content_loss *= content_weight
			
			# Compute TV Reg Loss
			tv_loss = torch.sum(torch.abs(blend_img[:, :, :, :-1] - blend_img[:, :, :, 1:])) + \
					torch.sum(torch.abs(blend_img[:, :, :-1, :] - blend_img[:, :, 1:, :]))
			tv_loss *= tv_weight

			# Compute Histogram Reg Loss
			hist_loss = 0

			match_time = time.process_time()

			for layer in range(0, len(blend_features_style)):
				with torch.no_grad():
					matched_features = get_matched_features_pytorch(blend_features_style[layer].detach(), target_features_style[layer])
				hist_loss += torch.norm(blend_features_style[layer] - matched_features, p="fro")

			match_time = time.process_time() - match_time

			hist_loss /= len(blend_features_style)
			hist_loss *= hist_weight
			
			# # l_psnr = psnr_weight * psnr(target_img / 255., input_img.clamp(0, 255.) / 255.)
			l_ssim = (1 - ssim(source_img * mask_img / 255., (input_img * mask_img).clamp(0, 255.) / 255.)) * ssim_weight
			# Compute Total Loss and Update Image
			loss = grad_loss + style_loss + content_loss + tv_loss + hist_loss + l_ssim
			optimizer.zero_grad()
			loss.backward()

			# Print Loss
			if run[0] % 1 == 0:
				print("run {}:".format(run))
				print('grad : {:4f}, style : {:4f}, content: {:4f}, tv: {:4f}, hist_loss: {:4f}, ssim_loss: {:4f}'.format(\
							grad_loss.item(), \
							style_loss.item(), \
							content_loss.item(), \
							tv_loss.item(), \
							hist_loss.item(), \
							l_ssim.item()
							))
				print()
			
			run[0] += 1
			return loss

		optimizer.step(closure)

	input_img.data.clamp_(0, 255)

	# Make the Final Blended Image
	blend_img = torch.zeros(target_img.shape).to(gpu_id)
	blend_img = input_img*canvas_mask + target_img*(canvas_mask-1)*(-1) 
	blend_img_np = blend_img.transpose(1,3).transpose(1,2).cpu().data.numpy()[0]

	# Save image from the first pass
	first_pass_img_file = os.path.join(opt.output_dir, background_img.split('/')[-1])
	imsave(first_pass_img_file, blend_img_np.astype(np.uint8))

def show_mask(mask, image, random_color=True):
	if random_color:
		color = np.concatenate([np.random.random(3), np.array([0.8])], axis=0)
	else:
		color = np.array([30/255, 144/255, 255/255, 0.6])
	h, w = mask.shape[-2:]
	mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)

	annotated_frame_pil = Image.fromarray(image).convert("RGBA")
	mask_image_pil = Image.fromarray((mask_image.cpu().numpy() * 255).astype(np.uint8)).convert("RGBA")

	return np.array(Image.alpha_composite(annotated_frame_pil, mask_image_pil)), mask_image

def get_mask_contour(mask_img):
	# Find contours in the mask image
	contours, _ = cv2.findContours(mask_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	# Assume the largest contour is the ID card
	contour = max(contours, key=cv2.contourArea)
	return contour, contours

def order_points(pts):
	rect = np.zeros((4, 2), dtype = "float32")

	s = pts.sum(axis = 1)
	rect[0] = pts[np.argmin(s)]
	rect[2] = pts[np.argmax(s)]

	diff = np.diff(pts, axis = 1)
	rect[1] = pts[np.argmin(diff)]
	rect[3] = pts[np.argmax(diff)]

	return rect

def perspective_transform(new_id_img, contour, mask_img):
	# Define the four corners of the new ID image
	h, w = new_id_img.shape[:2]
	pts1 = np.float32([[0, 0], [w, 0], [w, h], [0, h]])

	# Approximate the contour to a quadrilateral
	epsilon = 0.125 * cv2.arcLength(contour, True)
	approx = cv2.approxPolyDP(contour, epsilon, True)

	if len(approx) != 4:
		raise ValueError("The contour does not have exactly 4 points. Adjust the approximation parameters or ensure the mask is correct.")

	# Define the target points from the contour
	pts2 = order_points(np.float32([point[0] for point in approx]))

	# Get the perspective transformation matrix
	M = cv2.getPerspectiveTransform(pts1, pts2)

	# Apply perspective transformation
	transformed_id_img = cv2.warpPerspective(new_id_img, M, (mask_img.shape[1], mask_img.shape[0]))

	# increase the size of transformed id by 4% to cover the mask
	y1, x1 = transformed_id_img.shape[:2]
	transformed_id_img = cv2.resize(transformed_id_img, (0, 0), fx=1.04, fy=1.04, interpolation = cv2.INTER_CUBIC)
	y2, x2 = transformed_id_img.shape[:2]
	transformed_id_img = transformed_id_img[(y2-y1)//2+1: (y2+y1)//2+1, (x2-x1)//2: (x2+x1)//2]
	return transformed_id_img, M
 
def resize(image):
	resized_image = cv2.resize(image, (512, 512), interpolation=cv2.INTER_LANCZOS4)
	return resized_image

def replace_id_in_image(background_img_path, new_id_img_path, mask_image_1):
	os.makedirs(opt.background_dir, exist_ok=True)
	# Load images
	background_img = cv2.imread(background_img_path)
	new_id_img = cv2.imread(new_id_img_path)
	# print(mask_image[:, :, 0])
	mask_img = mask_image_1.cpu().numpy().astype(np.uint8)  # Load the mask as grayscale
	mask_img_1 = np.expand_dims(np.dot(mask_img[...,:3], [0.2989, 0.5870, 0.1140]), axis=2).astype(np.uint8)
	mask_img = np.where(mask_img>0, 255, mask_img)
	mask_img = mask_img[..., :3].astype(np.uint8)
	# print(mask_img_1.shape)
	# Get the contour from the mask
	contour, contours = get_mask_contour(mask_img_1)

	# Transform the new ID image to match the mask shape
	transformed_id_img, M = perspective_transform(new_id_img, contour, mask_img)

	# Resize the images
	mask_img = resize(mask_img)
	transformed_id_img = resize(transformed_id_img)
	background_img = resize(background_img)
	
	cv2.imwrite("mask_img.png", mask_img)
	cv2.imwrite("transformed_id_img.png", transformed_id_img)
	cv2.imwrite(os.path.join(opt.background_dir, background_img_path.split('/')[-1]), background_img)

	return "transformed_id_img.png", os.path.join(opt.background_dir, background_img_path.split('/')[-1]), "mask_img.png"

def main(background_img_path, new_id_img_path):
	TEXT_PROMPT = "id. document."
	BOX_TRESHOLD = 0.5
	TEXT_TRESHOLD = 0.25

	image_source, image = load_image(background_img_path)

	boxes, logits, phrases = predict(
		model=groundingdino_model,
		image=image,
		caption=TEXT_PROMPT,
		box_threshold=BOX_TRESHOLD,
		text_threshold=TEXT_TRESHOLD,
		device=DEVICE
	)
	print(boxes.shape)
	annotated_frame = annotate(image_source=image_source, boxes=boxes, logits=logits, phrases=phrases)
	annotated_frame = annotated_frame[...,::-1] # BGR to RGB

	sam_predictor.set_image(image_source)
	H, W, _ = image_source.shape
	boxes_xyxy = box_ops.box_cxcywh_to_xyxy(boxes) * torch.Tensor([W, H, W, H])
	transformed_boxes = sam_predictor.transform.apply_boxes_torch(boxes_xyxy, image_source.shape[:2]).to(DEVICE)
	masks, _, _ = sam_predictor.predict_torch(
				point_coords = None,
				point_labels = None,
				boxes = transformed_boxes,
				multimask_output = False,
			)
	global mask_image
	global annotated_frame_with_mask
	annotated_frame_with_mask, mask_image = show_mask(masks[0][0].cpu(), annotated_frame)

	# mask_image = torch.clamp(127.5 * mask_image + 128.0, 0, 255)
	# Replace the ID in the image
	transformed_id_image, background_img, mask_image = replace_id_in_image(background_img_path, new_id_img_path, mask_image * 255)
	run(transformed_id_image, background_img, mask_image)


background_img_path = 'midv_data/midv500/01_alb_id/images/CA/CA01_02.jpg'

new_id_img_path = 'IDNet/AZ.png'

main(background_img_path, new_id_img_path)
