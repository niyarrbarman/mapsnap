# MapSnap
## How to use
1. Clone the repository
2. Install the required packages
3. Run the script
<pre>
<code>git clone https://github.com/niyarrbarman/mapsnap.git
pip3 install -r requirements.txt
python3 app.py</code></pre>
## About MapSnap
MapSnap is a powerful tool designed to detect and map areas affected by landslides using satellite imagery and advanced deep learning technology. The Segformer model lies at the core of this technology, enabling MapSnap to analyze images pixel by pixel to accurately identify and segment affected areas.
The user-friendly interface allows emergency responders and disaster management teams to input satellite imagery and receive detailed reports in real-time, providing essential information for prompt and effective response to disasters. With MapSnap, disaster management teams can monitor affected areas and respond quickly to changes in disaster situations, improving overall response and recovery efforts.
MapSnap's cutting-edge technology makes it a valuable tool for disaster response and management. By leveraging the power of deep learning and satellite imagery, MapSnap can help to mitigate the impact of natural disasters, potentially saving lives and reducing the damage caused by landslides.

## About Segformers
Segformer is an advanced deep learning model that has been designed for binary segmentation tasks, such as identifying and segmenting objects in an image. This model is particularly well-suited for pixel-level image segmentation tasks, where the goal is to classify each pixel in an image as either foreground or background. The architecture of the Segformer model is based on the popular ViT (Vision Transformer) model, which has been adapted and modified to work for binary segmentation tasks.
One of the key advantages of the Segformer model is its ability to process large images with high accuracy, making it well-suited for satellite imagery analysis, such as in the case of MapSnap. By analyzing images pixel by pixel, Segformer can accurately detect the areas affected by natural disasters, such as landslides. This is particularly important in disaster response and management, where prompt action can be critical in saving lives and minimizing the impact of such events.
The Segformer model utilizes a self-attention mechanism, where each pixel is assigned a weight based on its relationship with the other pixels in the image. This helps the model to accurately identify and segment the foreground and background areas in the image. The model can also be fine-tuned for specific tasks, making it a versatile tool for a wide range of binary segmentation applications.
Overall, the Segformer model is a powerful tool for binary segmentation tasks, providing high accuracy and the ability to process large images with ease. Its use in MapSnap has revolutionized the way we respond to natural disasters, potentially saving lives and reducing the impact of these events. The Segformer model has many potential applications beyond disaster management, including medical imaging, object detection, and more, making it an important tool in the field of computer vision.
