 function sixth()
    warning('off');
    dataset=load('votes.txt');
    data=dataset(:,2:end);
    label=dataset(:,1);
    avg=[];
    for i=1:3
        avg(i)=linear_ridge(data,label,i);
    end
    fprintf('\nThe average error is %f',mean(avg));
    for i=1:3
        avg(i)=linear_kernel(data,label,i);
    end
    fprintf('\nThe average error is %f',mean(avg));
    for i=1:3
        avg(i)=polynomial_kernel(data,label,i);
    end
    fprintf('\nThe average error is %f',mean(avg));
    for i=1:3
        avg(i)=gaussian_kernel(data,label,i);
    end
    fprintf('\nThe average error is %f',mean(avg));
end