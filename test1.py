N=500
obs = numpy.random.binomial(1,0.2,N)
print(sum(obs)/500)
A = [[.3,.7],[.6,.4]]
B = [[.8,.2], [.1,.9]]
Pi = [.5,.5]
model = HMM(2,2,obs,A,Pi,B)
model.Baum_Welch()
model.show()