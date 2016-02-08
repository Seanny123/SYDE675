res = []
for d_l in dat_list:
    res.append(map_func(d_l))

all_dat = np.array(dat_list)
a_res = np.array(res)
a_res[a_res == 0] = -1
# grid the whole space
griddata()