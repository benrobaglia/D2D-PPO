# The algorithm is based on HAPPO: https://arxiv.org/pdf/2304.09870.pdf
# Agents are distributed and act according to their individual policy network.
# They can share information with their neighbours with D2D communications.
# The update is performed as follows: 
# 0) In an ACK/NACK, the BS informs the devices that the learning phase is going to happen in the next frame. 
# 1) The BS gathers all transitions (only observations) with D2D and computes A(s,a) with GAE.
# 2) They update their policy network sequentially and share their compound policy ratio to the next agent with D2D.
# 3) The BS updates the value network with MSE. 