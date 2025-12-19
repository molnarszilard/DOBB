### If an image does not contain any object that the methods should be trained on,
### it usually does not have a label.txt file for that image.
### Sometimes it is needed, therefore this script creates the empty label files for thoose 'clean' images.
### or moves or deletes the image without a label
### Maybe there is a label, but hte image for that label is deleted.
### use this scipt to delete the labels without images
import os
import argparse
import shutil

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--images', default="",
                    help='the directory to the images')
parser.add_argument('--labels', default="",
                    help='the directory to the labels')
parser.add_argument('--mode', default="dellabel",
                    help='which mode do you want? Options: addlabel (add empty labels if not exists), move (move images without labels), delimg (delete images without labels), dellabel (delete labels without images)')
args = parser.parse_args()

directory=args.images
images=[]
labels=[]
dlist=os.listdir(directory)
dlist.sort()
img_extensions=["png","jpg","tif"]
for filename in dlist:
    for iext in img_extensions:
        if filename.endswith(iext):
            #print(os.path.join(directory, filename))
            images.append(filename)
        else:
            continue
if len(images)<1:
    print("No images found.")
    exit()
directory=args.labels
dlist=os.listdir(directory)
dlist.sort()
for filename in dlist:
    if filename.endswith(".txt") and not filename.startswith("classes"):
        #print(os.path.join(directory, filename))
        labels.append(filename)
    else:
        continue

print("Number of images: "+str(len(images)))
print("Number of labels: "+str(len(labels)))

if args.mode in ['move']:
    folder = "empty/"
    isExist = os.path.exists(args.images+folder)
    if not isExist:
        os.makedirs(args.images+folder)

if args.mode in ['addlabel','delimg','move']:
    for i in range(len(images)):
        image_name=images[i]
        label_name = image_name[:-3]+"txt"
        print("Checking: "+label_name)
        if label_name not in labels:
            if args.mode in ['move']:
                print("Missing label. Moving image: %s"%(image_name))
                shutil.move(args.images+image_name,args.images+folder+image_name)
            elif args.mode in ['delimg']:
                print("Missing label. Deleting image: %s"%(image_name))
                os.remove(args.images+image_name)
            elif args.mode in ['addlabel']:
                print("Missing label. Adding empty label: %s"%(label_name))
                f = open(directory+label_name, "w")
                f.close()
elif args.mode in ['dellabel']:
    for i in range(len(labels)):
        label_name = labels[i]
        img = None
        for img_ext in img_extensions:
            if os.path.isfile(os.path.join(args.images, label_name[:-3]+img_ext)):
                image_name=label_name[:-3]+img_ext
                img = 0
        if img is None:
            print("Image %s.png/.jpg does not exists"%(label_name[:-3]))
            print("Missing image. Deleting label: %s"%(label_name))
            os.remove(args.labels+label_name)
            continue

