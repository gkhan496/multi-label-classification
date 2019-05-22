#dosya isimleri bir diziye atılacak
#dosya kadar dönüp her dosyayı gezecek

from tqdm import tqdm
import os

class Label_Processing:


    
    def __init__(self, *args, **kwargs):
        self.classes = []

        for i in os.listdir("data"):
            self.classes.append(i)
    
   
    def rename_for_label(self):
        for i in range(len(self.classes)):
            print(len(self.classes))
            train = input("Train_size for "+str(len(tqdm(os.listdir('data/'+self.classes[i]))))+"-"+self.classes[i]+" : ")
            
            for r,j in enumerate(tqdm(os.listdir('data/'+self.classes[i]))):
                if r < int(train):
                    src = 'data/'+self.classes[i]+'/'+j
                    dst = 'trainData/'+str(i)+"-"+j
                    os.rename(src,dst)
                else: 
                    src = 'data/'+self.classes[i]+'/'+j
                    dst = 'testData/'+str(i)+"-"+j
                    os.rename(src,dst)

                
    def information_for_labels():
        for i in range(len(classes)):
            for j in tqdm(os.listdir('train/'+classes[i])):
                file = open('information.txt','a+')
                file.write(classes[i]+"-->"+str(i))
                file.write('\n')
                break
#Label_Processing().rename_for_label()