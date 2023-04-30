# Notes




Dalton_DexlaPowderCalibration.py and Dalton_DexlaPower_Evaluation.py were copy/pasted from Dalton's github. they reference datafiles that can be found on dropbox. links to all those places can be foundin the converstaion below.




## Links from Dalton

Initial background on how to best do this came from a converstation with Dalton. Since Slack likes to delete thinks, here's a copy of the main points:


Dalton:
> Here's a link to [some ceria data and calibrated detector files](https://cornell.box.com/s/57xf3p9efnq3xontha4ma3e98yhekkvw) at different energies and different detector positions. Let me know if you have access issues. 

> I have put a [suite of scripts](https://github.com/daltonshadle/CHESS_hedmTools/tree/main/FF) on my repo for the dexela analysis . I have one script that takes in the 2 panel dexela config + ceria powder data and turns out chunked & calibrated 8 subpanel dexela config (DexelaPowderCalibration.py). I have another script for the bi-linear distortion analysis (DexelaPowderEvaluation.py). And I have another script I'm working on that compares different chunked & calibrated dexela config files from different experiments (DexelaComparison.py). These are all starting places more than anything, ie it's not great code, but it's better than nothing hopefully.

> On the CHESS side of things, I'm pushing for more ceria+multiruby datasets that were calibrated in the 2 panel configuration and hopefully we can quickly run the chunking & calibrating script to yield even more calibrated detector configs to fully characterize the relative position of the subpanels in each detector. I think once we hone in on the subpanel positions and plane normals, then we can hopefully turn it over to you guys to fix our distortion problem

https://github.com/daltonshadle/CHESS_hedmTools/tree/main/FF



Austin Gerlt
> sort of open ended question, but is there a good example of how to get eta/omega maps from ff-hexrd without using hexrdgui?


Dalton Shadle
>  https://github.com/daltonshadle/CHESS_hedmTools/blob/main/SingleGrainOrientationDistributions/hexrd3_dsgod/ReconstructDSGODs.py If you take the imports from this file and lines 565 to 708, you should have what you need. This uses the same guts as hexrdgui for constructing eta/omega maps, but in a command line or python script form. Feel free to adjust as you need!
