import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

phi=33.45*np.pi/180 #latitude of phoenix, AZ
df = pd.read_csv('PH722780_2000_solar.csv')
beta=phi
cosB = np.cos(beta)
param = (1-np.cos(beta))/2
parap = (1+np.cos(beta))/2
sinB = np.sin(beta)
rho = 0.2
#Initialize list
az = []
for i in range(21):
 az.append([])
#EPS is epsilon, the parameter that determines the clearness of the day.
eps = []
F = [[-0.008,0.130,0.330,0.568,0.873,1.132,1.060,0.678],[0.588,0.683,0.487,0.187,-0.392,-1.237,-1.6,-0.327],[-0.062,-0.151,-0.221,-0.295,-0.362,-0.412,-0.359,-0.250],[-0.06,-0.019,0.055,0.109,0.226,0.288,0.264,0.156],[0.072,0.066,-0.064,-0.152,0.462,-0.823,-1.127,-1.377],[-0.022,-0.029,-0.026,-0.014,0.001,0.056,0.131,0.251]]

 
#Create dates, hours, declinations and omegas. az[0] is day, numbered from 1-364(2000 was a leap year)
#az[1] is hour 0-23

for i in range(1, 367):
	for j in range(0,24):
		az[0].append(i) #day 
		az[1].append(j)#hour  
		az[2].append(np.pi*(23.45/180)*np.sin(2*np.pi*(284+i)/365)) #declination angle
		az[3].append((j-0.5-12)*np.pi/24) #omega- THIS SHOULD BE ALWAYS 0 for perfect tracking in horizontal axis?!

#calculate cosine values		
for i in range(len(az[0])):		
	az[5].append(1)#np.cos(phi-beta)*np.cos(az[2][i])*np.cos(az[3][i])+np.sin(phi-beta)*np.sin(az[2][i])) #costheta set to 1 for north-south tracking
	az[6].append(np.cos(phi)*np.cos(az[2][i])*np.cos(az[3][i])+np.sin(phi)*np.sin(az[2][i]))#coszenith
	az[15].append(np.arccos(az[6][i]))

		
#get cos(zenith) angles from csv and dump them into az[7] to compare with values calculated above
zenith = df['Zenith (deg)']
for i in range(0,len(zenith)):
	az[7].append(np.cos((np.pi/180.0)*zenith[i]))

#Need the hourly Extraterrestrial radiation, global, beam radiation and diffuse radiation
for i in range(0,len(az[0])):
	az[8].append(1367*(1+0.033*np.cos(2*np.pi*(az[0][i])/364))*az[6][i]) #I0
	if az[8][i] <0:
		az[8][i]=0
	az[9].append(df['METSTAT Glo (Wh/m^2)'][i]) #GLOBAL RADIATION
	az[10].append(df['METSTAT Dir (Wh/m^2)'][i]) #BEAM RADIATION
	az[11].append(df['METSTAT Dif (Wh/m^2)'][i]) #DIFFUSE RADIATION
	az[12].append(az[5][i]/az[6][i])	#Rb - parameter cos(theta)/cos(thetaZ)
	az[13].append(az[11][i]/(az[6][i]*az[8][i])) #DELTA
	az[19].append(((az[11][i]+az[10][i])/az[11][i] + 1.041*(az[15][i])**3)/(1+1.041*(az[15][i])**3)) # what is I?? this is epsilon

#Initialize clearness index

#for each epsilon value, we need to bin that into a clearness value, labelled az[20], which we cac then use to 
#find the F coefficients for the perez model.
F12=[[],[]]
for i in range(0,len(az[0])):
	if az[19][i] < 1.065:
		az[20].append(0)
	elif az[19][i] < 1.230:
		az[20].append(1)
	elif az[19][i] < 1.5:
		az[20].append(2)
	elif az[19][i] < 1.95:
		az[20].append(3)
	elif az[19][i] < 2.8:
		az[20].append(4)
	elif az[19][i] < 4.5:
		az[20].append(5)
	elif az[19][i] < 6.2:
		az[20].append(6)
	else:
		az[20].append(7)
	F12[0].append(F[0][az[20][i]]+F[1][az[20][i]]*az[13][i]+F[2][az[20][i]]*az[15][i]) #F1 coefficients
	F12[1].append(F[3][az[20][i]]+F[4][az[20][i]]*az[13][i]+F[5][az[20][i]]*az[15][i]) #F2 coefficients
	az[16].append(np.max([0,az[5][i]])) #a values
	az[17].append(np.max([0.087,az[6][i]])) #b values
	az[18].append(az[11][i]*((1-F12[0][i])*(1+cosB)/2 + F12[0][i]*(az[16][i]/az[17][i])+F12[1][i]*sinB)) #Perez Diffuse
	
sum=0	
# for i in range(0,len(az[0])):
	# az[14].append(np.absolute(az[10][i]*az[12][i]+az[11][i]*((1-az[13][i])*(parap)*(1+fmod*sinB)+az[13][i]*az[12][i])+az[9][i]*rho*param))
	# sum = sum+az[14][i]

	
# for i in range(len(az[0])):
		# if az[14][i] > 1000000000:
			# az[14][i] = 0
			# print i

plt.plot(az[1],az[18],'ro')
plt.show()

for i in range(1000,1300):
	print (az[20][i])#,az[7][i],az[10][i],az[8][i])#,az[11][15],az[12][15],az[1][15])
print sum

	
	
	
	
	
	
	
	
	
	
