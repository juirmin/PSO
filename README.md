# python pso 
![](https://img.shields.io/github/repo-size/juirmin/PSO)

![](https://img.shields.io/badge/Medium-Juirmin-black)

[![](https://avatars.githubusercontent.com/u/923954?s=200&v=4)](https://medium.com/@kmes990402081/python-%E5%AF%A6%E4%BD%9C-particle-swarm-optimization-pso-%E7%B2%92%E5%AD%90%E7%BE%A4%E6%9C%80%E4%BD%B3%E5%8C%96-ba173ed936ca "Python 實作 Particle Swarm Optimization,PSO(粒子群最佳化)")

▲Click the img to see more
## Particle Swarm Optimization
## 粒子群最佳化

```python
	class PSO():
		def __init__(self, pN, dim, max_iter):
			self.w = np.linspace(0.9, 0.4, max_iter)  # 慣性權重一般為 0.9~0.4 遞減
			self.c1 = 2  # 通常設為2
			self.c2 = 2  # 通常設為2
			self.pN = pN
			self.dim = dim
			self.max_iter = max_iter
			self.X = np.zeros((self.pN, self.dim))  # 粒子皆有自己的位置
			self.V = np.zeros((self.pN, self.dim))  # 粒子皆有自己的速度
			self.pbest = np.zeros((self.pN, self.dim))  # 粒子皆有自己的最佳位置
			self.pfit = np.zeros((self.pN))
			self.gbest = np.zeros((self.dim))  # 族群只有一個最佳位置
			self.fitness = 1e20

		def function(self, x):
			temporary = np.zeros((self.pN))  # 暫存
			for p in range(self.pN):
				fitness = 0
				for Qi, Qp in enumerate(Qx):
					t_fitness = 0
					for i, d in enumerate(x[p]):
						t_fitness += d*Qp**(self.dim-i-1)
					t_fitness -= Qy[Qi]
					fitness += t_fitness**2
				temporary[p] = fitness
			return temporary

		def update(self):
			now_t = self.function(self.X)
			for i, p in enumerate(now_t < self.pfit):  # 更新個體過去最佳
				if(p):
					self.pbest[i] = self.X[i]
			if(min(now_t) < self.fitness):  # 更新族群過去最佳
				index = np.where(now_t == min(now_t))[0][0]
				self.fitness = min(now_t)
				self.gbest = self.X[index]

		def init_Population(self):
			for i in range(self.pN):
				for j in range(self.dim):
					self.X[i][j] = random.uniform(-10, 10)
					self.V[i][j] = random.uniform(0, 5)
				self.pbest[i] = self.X[i]
			self.pfit = self.function(self.pbest)
			self.update()

		def iterator(self):
			fit = []
			self.init_Population()
			for iter in range(self.max_iter):
				self.V = self.w[iter]*self.V+self.c1*random.uniform(0.2, 0.7)*(
					self.pbest-self.X)+self.c2*random.uniform(0.2, 0.7)*(self.gbest-self.X)
				self.X = self.X+self.V
				self.update()
				print(self.fitness, self.gbest, iter)
				fit.append(self.fitness)
			return(fit)
 ```
