
function z = softthreshold(e1,e2 ,e3,x,y,r,spnum)  
z=zeros(spnum,1);
   z1=(x*e1+y*e2)/(x+y);
   z2=r/(2*(x+y));
for i=1:spnum
   if (z1(i,1) - e3(i,1) < -z2)
	s(i) = z1(i,1) + z2;
	else if (z1(i,1) - e3(i,1) > z2)
	s(i)= z1(i,1) - z2;
    else
	s(i) = e3(i,1); 
      end
   end
 z(i,1)=s(i);

end    