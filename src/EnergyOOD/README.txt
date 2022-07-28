The code provided for Energy OOD in https://github.com/wetliu/energy_ood is a mess and the paper is awful at explaining to someone who does not know thermal physics equations.
So I am going to try and find only what I need from the implementation and leave the rest.
As far as I can tell most of it is to get Mahalanobis and ODIN working (and setting up the model itself) and not actually for the energy based model.

But because I am not doing exactly everything they said, I may be missing something important, could someone check my work and make sure it matches up with what is needed?
I think I know what I need but... Well, I can't say I understand the paper well.


The "score" is how many times the algorithm assumes it is seeing something in distribution. So a higher number is better in tesing and lower is better when dealing with unknows.