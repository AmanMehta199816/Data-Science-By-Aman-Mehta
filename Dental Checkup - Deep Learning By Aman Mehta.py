#!/usr/bin/env python
# coding: utf-8

# # Synergy Project 2023 - Aman Mehta

# In[4]:


pip install tensorflow 


# In[2]:


import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers


# In[ ]:


image_height = 
image_width  = 
num_channels = 

num_classes  =  


# In[3]:


model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(image_height, image_width, num_channels)),
    layers.MaxPooling2D((2, 2)),
    # Add more convolutional and pooling layers here as needed
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(num_classes, activation='softmax')
])


# In[ ]:


model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit(training_images, training_labels, epochs=num_epochs, batch_size=batch_size,
          validation_data=(validation_images, validation_labels))


test_loss, test_accuracy = model.evaluate(test_images, test_labels)
print(f"Test Accuracy: {test_accuracy}")




# In[ ]:




