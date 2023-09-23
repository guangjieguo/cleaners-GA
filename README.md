# cleaners_GA
This project is an assignment for the Artificial Intelligence paper. For this assignment, 
I implemented a genetic algorithm to optimise the fitness of a
population of vacuuming agents, a.k.a. cleaners, tasked with cleaning an area. The rules
of the world are as follows: a cleaner can move forwards on backwards in discrete steps; it
can rotate in either direction by 90 degrees; when driving over dirty area it automatically
picks up the dirt (unless its bin is full); cleaners run on a battery that needs to be recharged
periodically at a charging station, where also the bin is emptied. The objective is clean
maximal area. Oh, and the final complication is that there is another population of cleaners
on the field, competing for cleaning credits. My algorithm should find behaviours that
lead to the most cleaning done by your cleaners.The engine is given by our lecturer Lech Szymanski. 
My jod is to build my_agent.py for implementingngenetic algorithm to obtain a population of
40 cleaners with good performance. 

For the detail of the algorithm and the evaluation, please see my_report.pdf. 
