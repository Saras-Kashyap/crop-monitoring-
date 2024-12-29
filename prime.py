import os
import cv2
import numpy as np



#data = pd.read_csv(r'D:\downlode.new\archive.zip\test\test')

# api = KaggleApi()
# api.authenticate()


# Path to your directory
#dir_path = r'C:\Users\SARAS KASHYAP\OneDrive\Desktop\crop1'

# # List all files with their full paths
# for root, dirs, files in os.walk(dir_path):
#     for file_name in files:
#         file_path = [os.path.join(root, file_name)]
#         print(file_path)
        

#todo wheat_img is decese image 
disease_str = [r"C:\Users\SARAS KASHYAP\program\finalprojet\diseases\leafspors3.png",r"C:\Users\SARAS KASHYAP\program\finalprojet\diseases\decleafspo2.png",r"C:\Users\SARAS KASHYAP\OneDrive\Desktop\dise crop\Anthracnose fruit rot (tomato).png",r"C:\Users\SARAS KASHYAP\OneDrive\Desktop\dise crop\bacterial leaf streak disease (rice).png",r"C:\Users\SARAS KASHYAP\OneDrive\Desktop\dise crop\Bacterial Wilt  (potato).png",r"C:\Users\SARAS KASHYAP\OneDrive\Desktop\dise crop\Early blight (tomato).png",r"C:\Users\SARAS KASHYAP\OneDrive\Desktop\dise crop\common rust(weat , rice).png",r"C:\Users\SARAS KASHYAP\program\finalprojet\diseases\Screenshot 2024-10-13 134859.png", r"C:\Users\SARAS KASHYAP\program\finalprojet\diseases\decleafspo.png" ,r"C:\Users\SARAS KASHYAP\program\finalprojet\diseases\opi scree.png",r"C:\Users\SARAS KASHYAP\program\finalprojet\diseases\dece.png",r"C:\Users\SARAS KASHYAP\program\finalprojet\diseases\decleafspo.png"]
for i in range(0 , len(disease_str)):

    #! farm_img is input image 
    farm_img = cv2.imread(r'C:\Users\SARAS KASHYAP\program\finalprojet\diseases\bacterial blight.png' , cv2.IMREAD_UNCHANGED)
    farm_img = cv2.cvtColor(farm_img, cv2.COLOR_BGR2GRAY)

    wheat_img = cv2.imread(disease_str[i] , cv2.IMREAD_UNCHANGED)
    wheat_img = cv2.cvtColor(wheat_img, cv2.COLOR_BGR2GRAY)
    
# cv2.matchTemplate, the images need to be of types CV_8U (8-bit) 
# or CV_32F (32-bit). Ensure farm_img and wheat_img meet these requirements
# farm_img = farm_img.astype(np.float32)
# wheat_img = wheat_img.astype(np.float32)
#? for showing farm_img
# cv2.imshow('Farm', farm_img)
# cv2.waitKey()
# cv2.destroyAllWindows()

    cv2.imshow('Needle', wheat_img)
    cv2.waitKey()
    cv2.destroyAllWindows()

# There are 6 comparison methods
# TM_CCOEFF
#? TM_CCOEFF_NORMED( we are using this alorithem)
# TM_CCORR
# TM_CCORR_NORMED
# TM_SQDIFF
# TM_SQDIFF_NORMED

#! (first orignal image then , the image we are matching)
    result = cv2.matchTemplate(farm_img, wheat_img, cv2.TM_CCOEFF_NORMED)
    cv2.imshow('result', result)
    cv2.waitKey()
    cv2.destroyAllWindows()

#todo for geting max_match , min_match ,value and  location
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

    print(max_loc)
    print(max_val)


# height and width of needal image 
    w = wheat_img.shape[1]
    h = wheat_img.shape[0]

#! for drawing rectangle
    cv2.rectangle(farm_img, max_loc, (max_loc[0] + w, max_loc[1] + h), (0,255,255), 2)
    print(cv2.rectangle(farm_img, max_loc, (max_loc[0] + w, max_loc[1] + h), (0,255,255), 2))

# cv2.imshow('Farm', farm_img)
# cv2.waitKey()
# cv2.destroyAllWindows()


    threshold = .90

    yloc, xloc = np.where(result >= threshold)
    print(len(xloc)) # number of match of neddal img

    for (x, y) in zip(xloc, yloc):
        cv2.rectangle(farm_img, (x, y), (x + w, y + h), (0,255,255), 2)

# cv2.imshow('Farm', farm_img)
# cv2.waitKey()
# cv2.destroyAllWindows()    

#! fro grouping the match boxes 
    rectangles = []
    for (x, y) in zip(xloc, yloc):
    # will duble the value of number of matchs
        rectangles.append([int(x), int(y), int(w), int(h)])
        rectangles.append([int(x), int(y), int(w), int(h)])
    print(disease_str[i]) #! with this line of codde we can identify match image
    print(len(rectangles))

    rectangles, weights = cv2.groupRectangles(rectangles, 1, 0.2)

    print(rectangles)

    print(len(rectangles))

    cv2.imshow('Farm', farm_img)
    cv2.waitKey()
    cv2.destroyAllWindows()    

    #! storing the acurcy of data/match
    MATCH_PER = max_val * 100
    print(MATCH_PER)

    #todo when we will ditect the disease we will break the loop 
    if MATCH_PER > 90:
        print("we have find the disease")
        print(disease_str[i])
        break;