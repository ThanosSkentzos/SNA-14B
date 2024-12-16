## Installation
After cloning the repo, to install our dependencies we used [pixi](https://github.com/prefix-dev/pixi) since it was the easiest way we found to install graph tool. To install on linux bash terminal emulator:
```bash
curl -fsSL https://pixi.sh/install.sh | bash
```
To install our dependencdies clone this project, enter the directory and run:
```bash
pixi update
```
To activate environment:
```bash
pixi shell
```
## How to use
We have included all the needed files to show the visualizations without need for recalculation. In case you want to test this on another dataset follow the procedure below:


- To run hierarchical core periphery enter the data folder and use the hcp executable in combination with the parameters file like so:
```bash
cd data
./hcp parameters_terror.txt
```
- To process the resulting data, run extract_hcp.py

- To run the related work algorithms run the related_work.py file. This will generate the data needed for visualizations.


- To run visualizations run the visualize.py file.