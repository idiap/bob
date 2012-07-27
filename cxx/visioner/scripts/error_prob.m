x=[0.0:0.1:10.0]; 
L=max(x);

p0=0.10;
beta=0.20;
a=(4*beta-(1-p0))/L; b=2*(1-p0-2*beta)/L/L;

pos=find(x>=0); L=max(x);
for i = pos, y(i)=p0+a*x(i)+b*x(i)*x(i); end;

plot(x,y);
