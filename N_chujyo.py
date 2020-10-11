#plot\scatter1import numpy as np
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import csv
import matplotlib

path='Boring.csv'
with open(path,encoding='utf-8') as f:
    f=csv.reader(f)
    all_csv=[row for row in f]
    csv_r=len(all_csv)
    csv_c=len(all_csv[0])

def pikup(list,position,new_list):
    for i in list[1:]:
        temp=float(i[position])
        new_list.append(temp)

d=[]
pikup(all_csv,0,d)
d=[i*-1 for i in d]

N=[]
pikup(all_csv,1,N)

print(N)
print(d)

fai=float(all_csv[1][4])
pile_head=float(all_csv[1][5])
pile_length=float(all_csv[1][6])
pile_tip=pile_head+pile_length

print(fai,pile_head,pile_length,pile_tip)

NU=(-1*fai/1000+pile_tip)*-1
NB=(fai/1000+pile_tip)*-1

NUB=list([NU,NB])
print(NUB)

for i in NUB:
    if i not in d:
        d.append(i)
        d.sort(reverse=True)
NUi=d.index(NU)
NBi=d.index(NB)

N.insert(NUi,0)
N.insert(NBi,0)

print(d)
print(N)
print(NUi,NBi)

#未確定N値位置
d_fast=round(abs(d[NUi+1]-d[NUi]),3)
d_last=round(abs(d[NBi]-d[NBi-1]),3)
print(d_fast,d_last)
#未確定N値
if d_fast<1:
    N1=(N[NUi+1]-N[NUi-1])*(d[NUi]-d[NUi-1])-N[NUi-1]
    N[NUi]=abs(round(N1,2))
if d_last<1:
    N2=(N[NBi+1]-N[NBi-1])*(d[NBi]-d[NBi-1])-N[NBi-1]
    N[NBi]=abs(round(N2,2))
print(N)
print('====')
sh=0
sa=0
for i in range(NUi,NBi):
    u=N[i]
    b=N[i+1]
    print(u,b)
    h=abs(d[i+1]-d[i])
    print(h)
    a=(u+b)*h/2
    sh=sh+h
    sa=sa+a
HN=sa/sh
print(round(HN,2))

N_cal=N[NUi:NBi+1]
d_cal=d[NUi:NBi+1]

print(NBi)
print(N_cal,d_cal)

ph=np.array([20,20,20+fai/400,20+fai/400])
pd=np.array([pile_head*-1,pile_tip*-1,pile_tip*-1,pile_head*-1])

#========================================
fig1=plt.figure(figsize=(3.2,7.2))
fig1.subplots_adjust(left=0.2, bottom=0.1, right=0.95,
                    top=0.95, wspace=0.15, hspace=0.15)
fig1.suptitle('柱状図')

ax1=fig1.add_subplot()

plt.xlabel('N_value')
plt.ylabel('depth')

plt.plot([0,20+fai/400],[NU,NU])
plt.plot([0,20+fai/400],[NB,NB])
plt.plot(ph,pd)
plt.plot(N,d,'bo',N,d,'k')

j=len(N)
print(len(N))

for i in range(0,j):
    ax1.annotate(N[i],xy=(N[i],d[i]),xytext=(N[i],d[i]))

#secoud graph
fig2=plt.figure(figsize=(2.7,2.7))
fig2.subplots_adjust(left=0.2, bottom=0.1, right=0.95,
                    top=0.95, wspace=0.15, hspace=0.15)
plt.xlabel('N_value')
plt.ylabel('depth')


ax2=fig2.add_subplot()

plt.plot(N_cal,d_cal,'bo',N_cal,d_cal,'k')

k=len(N_cal)
N_formular=[]
for i in range(0,k-1):
    N_formular.append('(')
    N_formular.append(N_cal[i])
    N_formular.append('+')
    N_formular.append(N_cal[i+1])
    N_formular.append(')*')
    i1=round(abs(d_cal[i+1]-d_cal[i]),3)
    N_formular.append(i1)
    N_formular.append('+')
del N_formular[-1]

N_for=[]
for i in N_formular:
    N_for.append(str(i))
type(N_for)
N_for=''.join(N_for)
a='N_bar=1/2*('
b= ')/'
c=str(abs(NB-NU))
d='='
e=str(round(HN,1))
N_for=a + N_for + b + c + d + e
print(N_for)

for i in range(0,k):
    ax2.annotate(N_cal[i],xy=(N_cal[i],d_cal[i]),xytext=(N_cal[i],d_cal[i]))
k1=max(N_cal)
plt.plot([0,k1],[NU,NU])
plt.plot([0,k1],[NB,NB])

ph1=np.array([20,20,20+fai/400,20+fai/400])
pd1=np.array([d_cal[0],pile_tip*-1,pile_tip*-1,d_cal[0]])
plt.plot(ph1,pd1)

ax2.annotate(NU,xy=(0,NU),xytext=(0,NU))
ax2.annotate(NB,xy=(0,NB),xytext=(0,NB))
p_t=pile_tip*-1
ax2.annotate(p_t,xy=(0,p_t),xytext=(0,p_t))
ax2.annotate(N_for,xy=(0,p_t),xytext=(0,p_t-0.1))
plt.plot([0,k1],[p_t,p_t])

print(pile_tip)
plt.show()
fig1.savefig("img1.png")
fig2.savefig("img2.png")

#=========================================
import openpyxl
from openpyxl.drawing.image import Image

from openpyxl import load_workbook
wb = load_workbook("柱状改良の検討.xlsx")
ws = wb["N"]
img1= openpyxl.drawing.image.Image('img1.png')
img2= openpyxl.drawing.image.Image('img2.png')
#img.anchor='c3'
#print(type(img))

ws.add_image(img1,'B3')
ws.add_image(img2,'G10')

wb.save("柱状改良の検討.xlsx")
