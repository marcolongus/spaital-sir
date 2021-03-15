#Animación de la epidemia a través de agentes. 

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as patches
import os

colores = ['blue','red', 'green', ]
archivo = "data/animacion.txt"

##############################################################################################
#Animacion
##############################################################################################
def trayectoria(ni_steps=0, np_steps=0, tpause=0.01):

	N=100
	L=55

	fig, ax = plt.subplots()

	for i in range(ni_steps,np_steps):

		if(i%100==0):
			print(i)

		x = np.loadtxt(archivo, usecols=0, skiprows=N*i, max_rows=N)
		y = np.loadtxt(archivo, usecols=1, skiprows=N*i, max_rows=N)

		estado = np.loadtxt(archivo, usecols=3, skiprows=N*i, max_rows=N, dtype=int)
				
		plt.cla()

		plt.title("Agents system") 
		plt.xlabel("x coordinate") 
		plt.ylabel("y coordinate")

		plt.axis('square')
		plt.grid()
		plt.xlim(-1,L+1)
		plt.ylim(-1,L+1)

		try:
			for j in range(N):
				circ = patches.Circle((x[j],y[j]), 1, alpha=0.7, fc= colores[estado[j]])
				ax.add_patch(circ)
		except:
			pass

		plt.savefig("video/pic%.4i.png" %(i), dpi=100)
		#plt.pause(tpause)

#########################################
def animacion(activada=False):
	if activada:
		path = "C:/Users/Admin/Desktop/Ident Evo-spatial-sir-CI. Dif inmu/video"
		print(os.getcwd())
		os.chdir(path)
		print(os.getcwd())
		os.system('cmd /k "ffmpeg -r 30 -f image2 -s 1920x1080 -i pic%04d.png -vcodec libx264 -crf 25  -pix_fmt yuv420p test.mp4"')

#########################################
def plot_data(activada=False):
	if activada:
		data = ["data/epid.txt"]
		maxrows = None
		N = 100

		sane       = np.loadtxt(data[0], usecols=0, max_rows=maxrows)
		infected   = np.loadtxt(data[0], usecols=1, max_rows=maxrows)
		refractary = np.loadtxt(data[0], usecols=2, max_rows=maxrows)
		time       = np.loadtxt(data[0], usecols=3, max_rows=maxrows)

		plt.xlabel("Time")
		plt.ylabel("Populations")
		plt.ylim(0,N)

		plt.axhline(y=0, color="black")
		plt.axvline(x=0, color="black")

		plt.plot(time, sane      , label = "Sane"      )
		plt.plot(time, infected  , label = "Infected"  )
		plt.plot(time, refractary, label = "Refractary")

		plt.legend()
		plt.grid()
		plt.show()

##############################################################################################
##############################################################################################


#trayectoria(12826,14131)
animacion(True)
plot_data()

