Пусть у нас есть $P_{in}=1, P_{out}=0$, матрица 3x2. Мы используем предположения из [[Результаты общения 10.04 с Василием]]. $L_{x}=3, L_{y}=2$
![[Pasted image 20260410152752.png]]

Я посчитал уравнения для каждого из элементов. В них можно подставить $h_{x},h_{y}$ и проверить корректность.

1 - $\Delta P_{00}=0 \to P_{0}\left( -\frac{2}{h_{x}^2}-\frac{1}{h_{y}^2} \right)+\frac{P_{1}}{h_{x}^2}+\frac{P_{3}}{h_{y}^2}=-\frac{P_{in}}{h_{x}^2}$
2 - $\Delta P_{10}=0 \to P_{1}\left( -\frac{2}{h_{x}^2}-\frac{1}{h_{y}^2} \right)+\frac{P_{2}}{h_{x}^2}+\frac{P_{0}}{h_{x}^2}+\frac{P_{4}}{h_{y}^2}=0$
3 - $\Delta P_{20}=0 \to P_{2}\left( -\frac{2}{h_{x}^2}-\frac{1}{h_{y}^2} \right)+\frac{P_{1}}{h_{x}^2}+\frac{P_{5}}{h_{y}^2}=-\frac{P_{out}}{h_{x}^2}$
4 -  $\Delta P_{01}=0 \to P_{3}\left( -\frac{2}{h_{x}^2}-\frac{1}{h_{y}^2} \right)+\frac{P_{4}}{h_{x}^2}+\frac{P_{0}}{h_{y}^2}=-\frac{P_{in}}{h_{x}^2}$
5 - $\Delta P_{11}=0 \to P_{4}\left( -\frac{2}{h_{x}^2}-\frac{1}{h_{y}^2} \right)+\frac{P_{5}}{h_{x}^2}+\frac{P_{3}}{h_{x}^2}+\frac{P_{1}}{h_{y}^2}$
6 - $\Delta P_{21}=0 \to P_{5}\left( -\frac{2}{h_{x}^2}-\frac{1}{h_{y}^2} \right)+\frac{P_{4}}{h_{x}^2}+\frac{P_{2}}{h_{y}^2}=-\frac{P_{out}}{h_{x}^2}$
