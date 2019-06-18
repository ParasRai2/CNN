# Pythono3 code to rename multiple  
# files in a directory or folder 
  
# importing os module 
import os 

DIR = "test"

# Function to rename multiple files 
def main(): 
    i = 0
      
    for filename in os.listdir(DIR): 
        dst = str(i) + ".jpg"
        src =DIR + '/'+ filename 
        dst =DIR + '/'+ dst 
          
        # rename() function will 
        # rename all the files 
        os.rename(src, dst) 
        i += 1
main()  
