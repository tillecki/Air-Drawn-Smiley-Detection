function [pos, vel] = acc2Imag(time,acc_x,acc_y, acc_z)
%ACC2IMAG Summary of this function goes here
%   Detailed explanation goes here
acc_x = acc_x - median(acc_x);
acc_y = acc_y - median(acc_y);
acc_z = acc_z - median(acc_z);
%acc_z = zeros(length(acc_z));


% figure;
% plot(time,acc_x);
% hold on;
% plot(time,acc_y);
% plot(time,acc_z);
% legend('X', 'Y', 'Z');
% xlabel('Relative time (s)');
% ylabel('Acceleration (m/s^2)');

% Integrate

%asd = findacc_x(~isnan(acc_x))
vel_x = cumtrapz(time,acc_x);
vel_x = vel_x - median(vel_x);
pos_x = cumtrapz(time,vel_x);
pos_x = pos_x - median(pos_x);
vel_y = cumtrapz(time,acc_y);
vel_y = vel_y - median(vel_y);
pos_y = cumtrapz(time,vel_y);
pos_y = pos_y - median(pos_y);
vel_z = cumtrapz(time,acc_z);
vel_z = vel_z - median(vel_z);
pos_z = cumtrapz(time,vel_z);
pos_z = pos_z - median(pos_z);



% figure
% hold on
% subplot(3,1,1)
% hold on
% plot(time, acc_x)
% plot(time, acc_y)
% plot(time, acc_z)
% title('Acc')
% legend('X', 'Y', 'Z');
% subplot(3,1,2)
% hold on
% plot(time, vel_x)
% plot(time, vel_y)
% plot(time, vel_z)
% title('Vel')
% subplot(3,1,3)
% hold on
% plot(time, pos_x)
% plot(time, pos_y)
% plot(time, pos_z)
% title('Dsp')
% 
% 
% figure;
% plot(pos_x,pos_y)
% xlim([-1 1])
% ylim([-1 1])
% 
% figure;
% plot3(pos_x,pos_y,pos_z)
% zlim([-1 1])
% xlim([-1 1])
% ylim([-1 1])
% 
% xlabel('x');
% ylabel('y');
% zlabel('z');
% 
pos = [pos_x,pos_y,pos_z];
vel = [vel_x, vel_y, vel_z];


end

