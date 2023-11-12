# Neural-style-transfer

In this assignment, the objective is to develop a deep learning model for artistic style transfer. The model is designed to analyze the unique aesthetic features of a chosen artwork and apply those characteristics to a new and original piece. The goal is to seamlessly integrate the artistic style of the selected work into the new artwork, making it appear as if the latter could have been created by the same artist. 
The implementation involves leveraging a pre-trained VGG19 model and employing TensorFlow and Keras to perform style transfer. VGG19 neural network to discern and extract features from both content and style layers of given artworks.The code defines functions for loading, preprocessing, and deprocessing images, along with content and style loss calculations based on feature representations. The core function, style_transfer, orchestrates the optimization process, adjusting a base image to minimize the differences in content and style features, ultimately generating a new artwork that embodies the stylistic essence of the selected art. 
The model optimizes a generated image to minimize the differences in content and style between the new artwork and the reference art, ultimately producing a visually coherent composition that captures the essence of the chosen artistic style. Parameters such as epochs, content_weight, and style_weight are customizable to fine-tune the adaptation process. The assignment addresses the challenging task of emulating artistic styles through deep learning, fostering creativity and generating unique artworks inspired by established artistic aesthetics.

# Input 


