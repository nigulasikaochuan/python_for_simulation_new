from scipy.io import loadmat

from vision_dom import vision
const = loadmat('./qam32_unit.mat')['qam32']

first_line = const[0:4,:]
end_line = const[-4:,:]

const2 = const[4:-4,:].reshape(6,4)


# print(const['qam32'])
print(first_line)
print(end_line)
print(const2)