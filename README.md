# EchoSign-KSL-Translator

Communication is a fundamental part of human life. It allows people to share ideas, express feelings, and build relationships. Yet, for deaf and hard-of-hearing individuals, communication is often limited by a gap that society has not fully bridged.

The application uses computer vision and machine learning techniques to recognize hand gestures and convert them into text and speech. It also processes spoken language and converts it into readable text, enabling smoother interaction.


##How It Works                                                                                                                                       
This system captures human communication through two intelligent pipelines: a camera-based vision module and a speech recognition module. 
Using MediaPipe, the application detects and tracks hand landmarks in real time, normalizes their spatial relationships, and feeds them into a trained machine learning model built with scikit-learn to accurately classify sign language gestures into letters and words. 
In parallel, spoken input is processed through Vosk, converting audio into text, which is then refined and mapped into visual sign representations. 
By combining computer vision, pattern recognition, and natural language processing, the system creates a seamless bridge between speech and sign language—transforming silent gestures into voice and spoken words into visual meaning—bringing communication closer, more inclusive, and profoundly human-centered.

##Introduction/Why i came up with this project                                                                                                       
There was a day when i went to competition in a school that had deaf students.The students had no other way of communication and socializing with us other than just writing on papers. Eventually, the papers got filled up and they had to search for another paper, even plucking them out of their books just to talk to us. During thier project showcase, they had to make sure they had a sign language translator. What if the translator was ubsent? How would they showcase their project and there was a big line of young inventors ready to showcase their project.  Most of the deaf/dumb (it was a mixed special school) students' projects were not related to technology. In my view, i knew they avoided technology projects because they need time to be fully understood.
I started thinking about how they learn sciences(Physics, Chemistry, Biology and maybe Computer studies). What if they did not have fully specialised / Good teachers that have a good background of teaching sciences well.

This made me think of how i can help them, which led me to think what if there was a device that captures signs then convert them to speech? This led me to think about how AI can be used to detect the sign language then convert it to speech. Thiss made a hearing individual communicate easily with a non-hearing person.What if there was a non-speaking person who wanted to communicate with a speaking person? I come up with a reversed version of the project which was SIGN-TO-SPEECH to SPEECH-TO-SIGN. This solved both problems.

##Features:                                                                                                                                 
It has features like :                                                                                                                  
 * 1.Letter recognition                                                                                                                       
 * 2.Word formation                                                                                                                            
 * 3.Conversation Logging                                                                                                                          
 * 4.Real_time hand detection                                                                                                                     
 * 5.Accuracy of 98%
 * 6.Confidence Filtering                                                                                                                            

##Development Process                                                                                                                                
The development of the EchoSign system followed an iterative and practical approach, focusing on building a functional and reliable communication tool rather than a theoretically perfect one. The process began with problem identification, where the need to bridge communication between speech users and sign language users was clearly defined. Based on this, the system was designed with two core components: a sign recognition module and a speech-to-sign translation module.

During implementation, the sign recognition module was developed using MediaPipe to detect and track hand landmarks in real time. These landmarks were processed and normalized to ensure consistency regardless of hand position or distance from the camera. A machine learning model was then trained using scikit-learn on manually collected data representing different hand signs. The model was integrated into a live prediction system, where additional techniques such as confidence filtering and temporal control were applied to reduce noise and improve stability.

In parallel, the speech recognition component was implemented using Vosk to convert spoken language into text. Since speech recognition is inherently imperfect, a lightweight correction mechanism based on approximate matching was introduced to improve the clarity of recognized words. This ensured that even when minor errors occurred, the system could still produce meaningful output.

The integration phase focused on combining both modules into a single graphical interface using Tkinter. The interface was designed to display recognized signs and detected speech clearly, while reducing dependency on terminal outputs. Additional features such as automatic text-to-speech feedback and conversation logging were included to enhance usability and traceability.

Throughout development, continuous testing and refinement played a key role. Practical challenges such as inconsistent lighting, repeated sign detection due to prolonged gestures, and occasional speech misrecognition were addressed through adjustments in thresholds, timing logic, and preprocessing techniques. While the system does not achieve perfect accuracy, it performs reliably under normal conditions and demonstrates a working solution that effectively bridges two modes of communication.

 ##Challenges Faced during the development and testing
  * Poor lighting affecting detection of hand landmarks
  * Similar shape/position/  causing confusion
  * Camera quality limitations
  * Misrecognization  of words

##Solutions Implemented
 * Improved lighting conditions
 * Reduced similar gesture classes
 * Added confidence filtering
 * Increased dataset size and variation

##Limitations
 * Only supports static gestures
 * Does not support input of motion-based signs (e.g., J, Z)
 * Limited vocabulary
 * Works best under good lighting
 * Sometimes slow
