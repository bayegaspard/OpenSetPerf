#These are all the functions that dont fit elsewhere
import Config


#Translation dictionaries for algorithms that cannot have gaps in their numbers.
relabel = {15:15}
rerelabel = {15:15}
temp = 0
for x in range(15):
    if temp < len(Config.helper_variables["unknowns_clss"]["unknowns"]) and x == Config.helper_variables["unknowns_clss"]["unknowns"][temp]:
        temp = temp+1
    else:
        relabel[x] = x-temp
        rerelabel[x-temp] = x
temp = None