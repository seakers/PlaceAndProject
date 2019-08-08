import numpy as np
import pandas as pd

""" make a sample problem:
ask pau for np.cost model
do something simple with orbital elements and revisit time.
    mean response time. Just use 
"""

def constellationCost12(t,np,h,i,massSat):
    '''Function that computes the approximate np.cost[M$] of a walker architecture given
    the number of satellites t and their respective altitude h.
    Author: Pau Garcia Buzzi
    translated from MATLAB by: Nathan Knerr
    '''

    #Constants
    G=6.673e-11
    mEarth=5.98e24
    rEarth=6378137
    WGS84_EARTH_MU=	3.986004418e14
    #Liquid hydrogen Impulse
    Isp=300
    #delta V required to go from 0km to a LEO of 400km and 28.7deg
    deltaV_ground_to_400=10000
    #delta V required for a Hohmann transfer orbit from 400km to h
    aL=rEarth+400*1000; aH=rEarth+h*1000; aT=0.5*(aL+aH)
    #burn at perigee
    deltaV1=np.sqrt(WGS84_EARTH_MU)*(np.sqrt(2/aL-1/aT)-np.sqrt(1/aL))
    #combined plane change with tangential brun at apogee
    Vi=np.sqrt(WGS84_EARTH_MU*(2/aH-1/aT))
    Vf=np.sqrt(G*mEarth/(aH))
    deltaV2=np.sqrt( Vf^2 + Vi^2 - 2*Vf*Vi*np.cos(np.radians(i-28.7)) )
    #Total delta V
    deltaV=deltaV_ground_to_400+abs(deltaV1)+abs(deltaV2)

    #$/kg_propellant
    cp=17
    Coffset=4.2567e+06
    #cp=5.21
    #Coffset=4.7722e+06

    #propellant mass computation
    mi_mf=np.exp(deltaV/9.8/Isp)
    ms=1250

    #cost Constellation
    S=90; B = 1 - ( np.log(100/S)/np.log(2) ); L=t^B
    np.costSatellite=1e3*(1064+35.5*massSat^1.261); np.costConstellation=np.costSatellite*L

    #Maximum payload Rocket labs Launch Vehicle
    if h==400:
        maxPayload=158.75
    elif h==500:
        maxPayload=150
    elif h==600:
        maxPayload=141.25
    elif h==700:
        maxPayload=132.5
    elif h==800:
        maxPayload=123.75

    #Dedicated launch
    mpayload=massSat*(t/np)
    Nded=np.ceil(mpayload/maxPayload)
    mp=mi_mf*(ms+mpayload/Nded)-ms-mpayload/Nded
    np.costLaunchVehicle=cp*mp; np.costLaunchVehicleTotal=np*(Coffset+np.costLaunchVehicle); %np.costLaunchVehicleTotal=np*np.costLaunchVehicle
    #np.costDedicated=Nded*(Coffset + np.costLaunchVehicleTotal) + np.costConstellation
    np.costDedicated=Nded*(np.costLaunchVehicleTotal) + np.costConstellation

    #Rideshare launch
    alpha=0.1
    mpayload1=massSat*(t/np)
    if mpayload1<= alpha*maxPayload:
        mpayload0=maxPayload*(1-alpha)
        mpayload=mpayload0+mpayload1
        mp=mi_mf*(ms+mpayload)-ms-mpayload
        mp0=mi_mf*(ms+mpayload0)-ms-mpayload0
        np.costLaunchVehicle=cp*mp-cp*mp0; np.costLaunchVehicleTotal=np*np.costLaunchVehicle
        np.costRideshare=np.costConstellation+np.costLaunchVehicleTotal
    else:
        np.costRideshare=100000000000000000000000000

    np.cost=min(np.costDedicated,np.costRideshare)/1e6

    if np.costDedicated>np.costRideshare:
        #rideshare
        ride_dedicated=1
    else:
        #dedicated
        ride_dedicated=0
    return np.cost, ride_dedicated
