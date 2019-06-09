import cv2
import numpy as np
import os



try: 
      
    # creating a folder named data 
    if not os.path.exists('imm'): 
        os.makedirs('imm') 
  
# if not created then raise error 
except OSError: 
    print ('Error: Creating directory of data') 
 
people = [person for person in os.listdir("imm/")]

for i in people:
	os.system('python3 AgeGender.py --input '+'imm/'+i)
	print(i)
