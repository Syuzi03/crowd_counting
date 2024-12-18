
import PIL.Image as Image
import numpy as np
from matplotlib import cm as c
import torch
from torchvision import transforms
import cv2


# convert image to a tensor and normalize
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# lodaing model, use cuda(GPU) 
model = CSRNet().to('cuda')
checkpoint = torch.load('weights.pth', map_location="cuda")
model.load_state_dict(checkpoint)


cap = cv2.VideoCapture("crowd2.mp4") 

fps = cap.get(cv2.CAP_PROP_FPS) # get frames per second (FPS) of video
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) # Get width of each frame of video
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) # Get height of each frame of video

# Object for result video
out = cv2.VideoWriter('output2_video.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # transform current frame to PIL Image that our model can understand
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    # transform image to tensor and sent to GPU(if available)
    img_tensor = transform(img).unsqueeze(0).to('cuda')

    # model prediction
    output = model(img_tensor)

    # crowd count prediction
    predicted_count = int(output.detach().cpu().sum().numpy())

    # show prediction in video
    cv2.putText(frame, f"Predicted Count: {predicted_count}", (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

    
    # transform model to NumPy array of density map 
    temp = np.asarray(output.detach().cpu().reshape(output.detach().cpu().shape[2], output.detach().cpu().shape[3]))

    # normalize min and max to make visualization stabilize 
    temp = temp - np.min(temp)
    temp = temp / np.max(temp)  
    
    gamma = 0.4
    density_map = np.power(temp, gamma) * 255
    density_map = np.uint8(density_map)

    density_map = np.uint8(c.jet(density_map / 255.0) * 255)  # transform density map to color map
    density_map = cv2.cvtColor(density_map, cv2.COLOR_RGBA2BGR)  # transform to BGR for visualization

    # scale the density map to frame size
    density_map_resized = cv2.resize(density_map, (frame_width, frame_height))

    # apply a density map to the frame
    frame_with_density = cv2.addWeighted(frame, 0.5, density_map_resized, 0.7, 0)

    # write the processed frame to the output file
    out.write(frame_with_density)

# release resources (close video and capture)
cap.release()
out.release()
