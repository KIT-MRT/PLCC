# PLCC
Photometric laser-scanner to camera calibration (PLCC)

Calibrate a laser-scanner to a camera with a structure-free technique.
The sensors are registered using the laser spots that are visible in the camera image.
Since we do not need reconstruction of the scene, this method is particuarly interesting for low-resolution laser-scanners.
The data attached is taken from a Pepperl&Fuchs r2000, by tuning its horizontal resolution to 2°. 
The minimum number of laser rays of the scanner is 2.

## Installation

Non-standard dependencies : math3d, scipy, cv2 (all available in "pip install")  
Standard dependencies: glob, os, optparse, pickle  

## Usage

Point clouds and images are stored in the same directory (hereby called input_dir), each one named by 
timestamp_ns.csv or timestamp_ns.png (f.e. 1465281413942525161.csv)

You need point clouds in the following format:  
1465281413942525161.csv:  
X, Y, Z, Amplitude, TimeStamp  
3.40282346639e+38, 3.40282346639e+38, 3.40282346639e+38, -1, 1465281266463374068  
3.40282346639e+38, 3.40282346639e+38, 3.40282346639e+38, -1, 1465281266463374068  
-0.000296705665418, 0.118999630108, 0, 206, 1465281266463374068  
-0.000273018725084, 0.0729994894556, 0, 225, 1465281266463374068  
-0.000314157963342, 0.0629992166997, 0, 221, 1465281266463374068  
-0.00043009620905, 0.0689986595323, 0, 224, 1465281266463374068  

In the current implementation the scanning planes are in x-y direction (also several planes possible tilted around x or y)


You need !!undistorted!! images without background illumination in which the infra-red laser-scanner spots are visible.
Therefore, do not use infra-red filters in your lense.
During the data acquisistion use a card board or something similar to generate measurements at various differences from the laser-scanner-camera rig.
Currently the outlier rejection is only supported for pinhole camera models. However the error minimized is also suited for wide-angle camera. In the examples we use lenses with 110° viewing angle.
Save the pinhole camera paramters in a textfile:
pinhole_calib.txt:  
f, cu, cv  
356.058898926, 456, 290  


Steps to do: 
* python sum_images.py input_dir path_to_summed_image.png
* cd output_dir
* python calibrate_PLCC.py -i input_dir  -o ./ -n path_to_pinhole_params.txt -a path_to_summed_image.png

For more plots:
* python do_plotting_laserscalib.py -i output_dir/Results.p -o figure_output_dir -e ".svg"

For evaluation we have the same conventions, do as described in "Photometric laser scanner to camera calibration (submittted to ITSC 2016)  

Execute:  
python evaluate_calibration_box_PCLL.py -r output_dir/Results.p -o output_dir -i eval_input_dir  

## Sample data

[sample_data] https://www.mrt.kit.edu/graeterweb/PLCC_sample_data.zip

## Contributing

1. Fork it!
2. Create your feature branch: `git checkout -b my-new-feature`
3. Commit your changes: `git commit -am 'Add some feature'`
4. Push to the branch: `git push origin my-new-feature`
5. Submit a pull request :D

## History

## Credits

## Related Papers

* **Photometric laser scanner to camera calibration**, *J. Gräter, T. Strauß, M.Lauer*, submittted to ITSC '16

## License

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
