# %% [markdown]
# # Inpainting Project

# %% [markdown]
# importing libraries we are going to need :

# %%
import imageio as iio
import numpy as np
import skimage.morphology as morpho  
from skimage import img_as_float
import matplotlib.pyplot as plt

# %% [markdown]
# ### Image Reading



# %%
def view(data, size=(10, 10), dpi=100):
    """
    Image Dispaly
    """
    fig = plt.figure()
    fig.set_size_inches(size)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.imshow(data)
    plt.show()

# %% [markdown]
# ### Rectangular mask creating function

# %%
def mask(im, x1, x2 , y1 , y2):
    """
    Takes into argument image , and four coodinates and
    produces mask that has image size 
    """
    shape = im.shape
    mask = np.ones((shape[0],shape[1]), dtype=int)
    for i in range(x1,x2):
        for j in range(y1,y2):
            mask[i,j]=0
    return mask

def read_mask(filename):
    return iio.imread(filename,as_grey=True)

# %%
def delete_zone(im,mask):
    """
    Takes as argument image and mask and produces image with
    nullified zone 
    """
    n,m,k=im.shape
    new_im = np.zeros((n,m,k),dtype=int)
    for i in range(3):
        new_im[:,:,i]=im[:,:,i]*mask
    return new_im

# %% [markdown]
# ## Patch manipulations :

# %%
def get_patch(image,p,patch_size=8):
    """
    Returns a patch centered on p
    """
    r = patch_size//2
    clip = np.array(image[p[0]-r:p[0]+r+1,p[1]-r:p[1]+r+1])
    return clip 


def similarity(patch1 , patch2 , maskpatch):
    d=0
    for i in range(3): 
        d+= np.sum(maskpatch*(patch1[:,:,i]-patch2[:,:,i])**2)
    return d

# %% [markdown]
# # Let's calculate the priority term

# %% [markdown]
# ## Confidence term

# %%
def c_matrix(mask):
    n,m= mask.shape
    c=np.zeros(mask.shape)
    for k in range(n):
        for l in range(m):
            patch = get_patch(mask,(k,l))
            c[k,l]=np.sum(patch)/(patch.shape[0]*patch.shape[1])
    return c

#view(c_matrix(mask(im, 10, 100 , 20, 200)))



# %%
#Bords du mask rectangulaire
# We need to find out how to do it in general case 
## Border using morphology
def init_bord_m(mask):
    n,m=mask.shape
    strell=morpho.disk(1)
    bords= mask-morpho.erosion(mask,strell)
    L=[]
    for i in range(n):
        for j in range(m):
            if(bords[i,j]==1):
                L.append((i,j))
    return L

def bord_matrix(mask,bord):
    """
    frontiere in a matrix ( for display purpuses only)
    """
    n,m=mask.shape
    bord_matrix=np.zeros((n,m))
    for i in range(n):
        for j in range(m):
            if((i,j) in bord):
                bord_matrix[i,j]=1
    return bord_matrix



# %%
def gradient_I(image ,mask, bordure):
        """
        We compute I(p) for all p in delta(\Omega)
        """
        h, w = image.shape[:2]
        c=image.copy()
        c_r= img_as_float(c[:,:,0])
        c_g= img_as_float(c[:,:,1])
        c_b= img_as_float(c[:,:,2])
        # We fill the outside of the mask with nones
        c_r[mask == 0] = np.NaN
        c_g[mask == 0] = np.NaN
        c_b[mask == 0] = np.NaN
        # We compute the gradient 
        fgradx,fgrady = np.array([np.zeros([h, w]),np.zeros([h, w])])
        for point in bordure:
            patch_r=get_patch(c_r,point)
            patch_g=get_patch(c_g,point)
            patch_b=get_patch(c_b,point)
            
            gradientR = np.nan_to_num(np.gradient(patch_r))
            gradientG = np.nan_to_num(np.gradient(patch_g))
            gradientB = np.nan_to_num(np.gradient(patch_b))
            
            normeR= np.sqrt(gradientR[0]**2 + gradientR[1]**2)
            normeG= np.sqrt(gradientG[0]**2 + gradientG[1]**2)
            normeB= np.sqrt(gradientB[0]**2 + gradientB[1]**2)
            
            norme = np.maximum(normeR,normeG,normeB)
     
            max_patch = np.unravel_index(
                norme.argmax(),
                norme.shape
            )
            fgradx[point[0], point[1]] = gradientR[0][max_patch]
            fgrady[point[0], point[1]] = gradientR[1][max_patch]

        return [fgradx,fgrady]
    

# %%
def normal_vect(image,mask,bord) : #n(p)
    h, w = mask.shape[:2]
    coordx , coordy= np.zeros((h, w)),np.zeros((h, w))
    for p in bord:
        i,j=p
        patch = get_patch(mask,(i,j))
        grad = len(patch)*len(patch[0])*np.nan_to_num(np.array(np.gradient(patch)))
        gradX = grad[0]
        gradY = grad[1]
        centerX, centerY = patch.shape[0]//2 ,patch.shape[1]//2
        coordx[i][j] =gradX[centerX][centerY]
        coordy[i][j] =gradY[centerX][centerY]
    return coordy,coordx


# %%
def P(image,mask,bordure):
    
    "Computes P for points of the bordure"
    
    h, w = mask.shape[:2]
    P=np.zeros((h,w))
    C=c_matrix(mask)
    I=gradient_I(image ,mask, bordure)
    N=normal_vect(image,mask,bordure)
    for (i,j) in bordure:
        P[i][j]=np.abs(I[0][i][j]*N[0][i][j]+
            I[1][i][j]*N[1][i][j])/255 * C[i][j]
    return P

def maxP(image,mask,bordure):
    
    "Finds point with max value of P"
    
    p=P(image,mask,bordure)
    maximum=p[bordure[0]]
    argmax=bordure[0]
    for point in bordure:
        i,j=point
        if(p[i][j]>=maximum):
            maximum = p[i][j]
            argmax=point
    return argmax  



# %% [markdown]
# # Iteration Of the algorithm

# %%

from tkinter import *

def iterate(image , firstmask, self , patch_size = 8 ,make_gif=True):
    print("Ha lghder bda")
    mask=np.copy(firstmask)
    print("sort le ts")

    frt = init_bord_m(mask)
    
    new_image= delete_zone(image,mask)
    #view(new_image)
    K=0
    ps=patch_size//2
    print(len(frt))
    while (len(frt)>0):
        print("le j c'est le s")
        p_point= maxP(new_image,mask,frt)
        print("sort le ts")
        p_patch=get_patch(new_image,p_point)
        new_patch = get_patch(new_image, (ps,ps))
        print("le j c'est le s")

        d = similarity(new_patch,p_patch,get_patch(mask,(ps,ps)))
        print(d)
        chosenX,chosenY = p_point
        print("point chosen is: {0},{1}".format(chosenX,chosenY))
        maskpatch = get_patch(mask,(chosenX,chosenY))
        # Looking for a patch that is the closest to the content of uncomplete patch
        for x in range(ps, new_image.shape[0]-ps):
            for y in range(ps,new_image.shape[1]-ps):
                potential = True
                firstpatch=get_patch(firstmask,(x,y))
                for i in range(patch_size):
                    for j in range(patch_size):
                        if(firstpatch[i,j]==0):
                            potential=False
                if (potential):
                    testPatch = get_patch(new_image,(x,y))
                    dtest = similarity(p_patch,testPatch,maskpatch)
                    if dtest < d :
                        d = dtest
                        print(dtest)
                        print('coordinates are {0} {1}'.format(x,y))
                        new_patch = np.copy(testPatch)
        #view(new_patch)
        # Filling
        for i in range(-ps,ps+1) :
            for j in range(-ps,ps+1):
                if(mask[chosenX+i,chosenY+j]==0):
                    new_image[chosenX+i,chosenY+j]= new_patch[ps+i,ps+j]
                    mask[chosenX+i,chosenY+j]= 1
        frt=init_bord_m(mask)
        #view(new_image)


        iio.imwrite("output/"+str(K)+".png", new_image)
        print(len(frt))
        self.d = Canvas(self.root, bg='white', width=self.sizey, height=self.sizex,relief=RIDGE,borderwidth=0)
        self.d.place(x=100,y=0)
        bg = PhotoImage(file = "output/"+str(K)+'.png')
        self.d .create_image( 0, 0, image = bg, anchor = "nw")
        self.d.update_idletasks()
        K+=1

    self.d = Canvas(self.root, bg='white', width=self.sizey, height=self.sizex,relief=RIDGE,borderwidth=0)
    self.d.place(x=100,y=0)
    bg = PhotoImage(file = "output/"+str(K)+'.png')
    self.d .create_image( 0, 0, image = bg, anchor = "nw")
    self.d.update_idletasks()
    

# %%


# %%


# %%



