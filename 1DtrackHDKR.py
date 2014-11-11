import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

phi=33.45*np.pi/180 #latitude of phoenix, AZ
df = pd.read_csv('PH722780_2000_solar.csv')
beta=phi
param = (1-np.cos(beta))/2
parap = (1+np.cos(beta))/2
sinB = (np.sin(beta/2))**3
rho = 0.2
#Initialize list
az = []
for i in range(20):
 az.append([])

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
	az[13].append(np.absolute(az[10][i]/az[8][i]))  #Anisotropy index (Ib/I0)

fmod = np.sqrt(az[10][i]/az[9][i])
sum=0	
for i in range(0,len(az[0])):
	az[14].append(np.absolute(az[10][i]*az[12][i]+az[11][i]*((1-az[13][i])*(parap)*(1+fmod*sinB)+az[13][i]*az[12][i])+az[9][i]*rho*param))
	sum = sum+az[14][i]

	
# for i in range(len(az[0])):
		# if az[14][i] > 1000000000:
			# az[14][i] = 0
			# print i

plt.plot(az[1],az[14],'ro')
plt.show()

#for i in range(1000,1300):
#	print (az[6][i],az[7][i],az[10][i],az[8][i])#,az[11][15],az[12][15],az[1][15])
print sum

	
	
	
	
	
	
	
	
	
	
