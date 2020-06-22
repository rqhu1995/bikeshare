# Model 1: Only User's demand are considered

Optimized using model provided in paper [综合考量借还车需求与调度成本的公共自行车调度优化模型.pdf](https://dl3.pushbulletusercontent.com/sAfuacFivpas1rKHCnVtl4aoqVgnK56d/综合考量借还车需求与调度成本的公共自行车调度优化模型.pdf)

<img src="https://dl3.pushbulletusercontent.com/Kn4IpJUiALil03i7bScfbH3O6vPZtywz/image.png" alt="image-20200620215455778" style="zoom:50%;" />

object (2) is dismissed for now

## encoding

- bike amount at each station as a vector, population of 300 at first.

- constraint <img src="/tex/d91a23306e3407b3eb0a4ac5cc26e0a0.svg?invert_in_darkmode&sanitize=true" align=middle width=81.53435399999998pt height=41.14169729999998pt/>
- constraint <img src="/tex/a8424a5eb296c448010bb08bd948ce8d.svg?invert_in_darkmode&sanitize=true" align=middle width=46.34118884999999pt height=22.831056599999986pt/> where <img src="/tex/bb29cf3d0decad4c2df62b08fbcb2d23.svg?invert_in_darkmode&sanitize=true" align=middle width=9.55577369999999pt height=22.831056599999986pt/> is the number of docks at each station 

## selection

tournament only

## crossover

single crossover, with probability of 0.5

## mutation

single mutation, with probability of 0.01

## result

best optimization i.e. fitness value is 502 (**in theory**, run with [Minizinc](https://www.minizinc.org/) with solver [Google OR-tools](https://developers.google.com/optimization/))

GA also generate solution with 502, with frequency of once in every 20 times.

Plot of fitness value change according to iteration time (example with solution 502):

<img src="https://dl3.pushbulletusercontent.com/fSY3ZsmzaSqeRncClTkIqVtIHwOU2SP1/image.png" alt="image-20200620220422959" style="zoom:50%;" />

# Model 2: Consider Truck Scheduling 

TBA...