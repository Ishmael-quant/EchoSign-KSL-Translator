# EchoSign-KSL-Translator

This is project that uses AI  to help others understand sign language. And in turn communicate with the deaf.
This project presents an Artificial Intelligence (AI) system that translates hand signs into text and speech in real time, helping bridge the communication gap.
This project uses a  system that enables **two-way communication** by translating:
* Hand signs → Text → Speech
* Speech → Text → Sign representations

Communication between deaf and hearing individuals is often limited by the lack of a common language. Sign language is widely used by deaf individuals, but many people do not understand it.

##How It Works
This system captures human communication through two intelligent pipelines: a camera-based vision module and a speech recognition module. 
Using MediaPipe, the application detects and tracks hand landmarks in real time, normalizes their spatial relationships, and feeds them into a trained machine learning model built with scikit-learn to accurately classify sign language gestures into letters and words. 
In parallel, spoken input is processed through Vosk, converting audio into text, which is then refined and mapped into visual sign representations. 
By combining computer vision, pattern recognition, and natural language processing, the system creates a seamless bridge between speech and sign language—transforming silent gestures into voice and spoken words into visual meaning—bringing communication closer, more inclusive, and profoundly human-centered.

##Features:
It has features like :
 1.Letter recognition
 2.Word formation
 3.Conversation Logging
 4.Real_time hand detection
 5.Accuracy of 98%

 ##Challenges Faced
  * Poor lighting affecting detection
  * Similar gestures causing confusion
  * Camera quality limitations

##Solutions Implemented
 * Improved lighting conditions
 * Reduced similar gesture classes
 * Added confidence filtering
 * Increased dataset size and variation

##Limitations
 * Only supports static gestures
 * Does not support motion-based signs (e.g., J, Z)
 * Limited vocabulary
 * Works best under good lighting
