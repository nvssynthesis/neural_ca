neural_ca is an interactive neural cellular automaton. Neural cellular automaton are akin to a classic cellular automaton, with the modifications:

* the states are effectively continuous (in actuality, floating-point)

* the transition rules are given by:

    - convolving the state grid with a convolution kernel 

    - applying a (usually nonlinear) activation function to the convolution result

While neural cellular automaton can be used very cleverly to solve real problems or simulate/model complex systems, in this project, the user is just meant to explore some of the possibilities of this system in an interactive manner.



neural_ca is essentially a game with no goal; you can play but you cannot win or lose. Here are the gameplay instructions:

* You can **enliven** cells by clicking on them, randomizing its color and giving it nonzero value. Of course, if the environmental conditions are such that the cell won't survive (such as the condition on startup), you should just see the cell light up for one or a few frames and then dissipate. If, on the other hand, the conditions around that cell allow for thriving, it should stay lit, or possibly transfer its life to a cell in its neighborhood. 

* You can **clear** the grid by hitting the space bar.

* You can **randomize** the grid by hitting the 'r' key.

<u>You will probably spend most of the gameplay with steering:</u>

* You can **steer**/navigate/manipulate the conditions (and thus the cells) by changing the 3x3 convolution kernel. To do this, the letter keys in the left of a qwerty keyboard are interpreted as the indices the kernel: 

  <div style="display: grid; grid-template-columns: repeat(3, auto); gap: 15px; text-align: left; padding: 10; justify-content: start;">
  <div>q</div> <div>w</div> <div>e</div>
  <div>a</div> <div>s</div> <div>d</div>
  <div>z</div> <div>x</div> <div>c</div>
  <p></p>
</div>

  * If you hold any of these keys while pressing the UP or DOWN arrows, you will increment or decrement the corresponding matrix kernel entry by a small amount.

* You can also **load** and **save** up to 10 kernel presets. If you click a number key, it loads a preset saved to that preset. If you hold shift while clicking a number key, it saves the current kernel settings to that preset. The game comes with 10 presets already, but you can overwrite them during the game. The presets are currently not saved between games; they return to the default set of presets.

* There is also a **terrain** hidden beneath the game surface. By default, it is completely hidden and has no effect. If you hold the 't' key and use up arrow, it will be revealed. To hide it again, decrement the value by holding 't' and the down arrow. The terrain affects the gradient of the environmental grid, leading to regions of more-or-less difficulty for cell survival. 

* You can also change the activation function by selecting a new option in the dropdown menu. A few notes about activation functions:

  * The cell values can overflow with unbounded functions. So, linear is not truly linear, it's more like a 'wrap'.
  * Some activations can cause high-frequency oscillations, meaning that areas of frames with switch between being bright and dark every other frame. The most clear example is COS_PI, which does y=cos(x\*pi). I do not find the extreme oscillating activation functions to be the most interesting, but left it in there for experimentation anyway. 