for i in range(data.shape[0]):                                                     #fill in missing data, data shape (5000, horizon, agent_num, 2) <- 2 stands for pos_x, pos_y
    
    if (i%100)==0:
        print(i)
    for j in range(data.shape[2]):
        left = [-1,-1]
        right = [-1,-1]
        left_index = 0
        right_index = 0
        index = 0

        while index < data.shape[1]:
            current_x = data[i,index,j,0]
            current_y = data[i,index,j,1]
            if np.isnan(current_x) and np.isnan(current_y):                                    #current timestep is missing, only update right_index
                index = index + 1
                right_index = right_index + 1
                continue
            if left_index<right_index:                                               #current timestep is not missing, but has previous missing timestep
                right = [current_x, current_y]

                if left[0] == -1 and left[1] == -1:                                  #missing timestep at the beginning, fill in first value
                    for x in range(left_index, right_index):
                        data[i,x,j,0] = right[0]
                        data[i,x,j,1] = right[1]
                else:
                    increment = ((right[0]-left[0])/(right_index-left_index),        #missing timestep in the middle, interpolation
                                    (right[1]-left[1])/(right_index-left_index))
                    start = left
                    for x in range(left_index, right_index):                            
                        data[i,x,j,0] = start[0]
                        data[i,x,j,1] = start[1]
                        start[0] += increment[0]
                        start[1] += increment[1]
                right_index = right_index + 1
                left_index = right_index
                left = right
                index += 1
                continue
            left = [current_x, current_y]                                          #current timestep is not missing and no previous missing timestep
            right = [current_x, current_y]
            left_index += 1
            right_index += 1
            index += 1
        if left_index<right_index:                                                 
            if (left[0] == -1 and left[1] == -1):                                  #if the whole traj is missing, remove entire traj
                data[i,:,j,:] = np.ones((data.shape[1],4))*(-1e10)