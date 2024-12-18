import PIL.Image as Image
import numpy as np
from matplotlib import cm as c
import torch
from torchvision import transforms
import cv2


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

model = CSRNet().to('cuda')
checkpoint = torch.load('weights.pth', map_location="cuda")
model.load_state_dict(checkpoint)


cap = cv2.VideoCapture("crowd2.mp4") 

fps = cap.get(cv2.CAP_PROP_FPS)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

out = cv2.VideoWriter('output2_video.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    img_tensor = transform(img).unsqueeze(0).to('cuda')

    output = model(img_tensor)

    predicted_count = int(output.detach().cpu().sum().numpy())

    cv2.putText(frame, f"Predicted Count: {predicted_count}", (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

    
    temp = np.asarray(output.detach().cpu().reshape(output.detach().cpu().shape[2], output.detach().cpu().shape[3]))

    temp = temp - np.min(temp)
    temp = temp / np.max(temp)  
    
    gamma = 0.4
    density_map = np.power(temp, gamma) * 255
    density_map = np.uint8(density_map)

    density_map = np.uint8(c.jet(density_map / 255.0) * 255)  
    density_map = cv2.cvtColor(density_map, cv2.COLOR_RGBA2BGR) 

    density_map_resized = cv2.resize(density_map, (frame_width, frame_height))

    frame_with_density = cv2.addWeighted(frame, 0.5, density_map_resized, 0.7, 0)

    out.write(frame_with_density)

cap.release()
out.release()
