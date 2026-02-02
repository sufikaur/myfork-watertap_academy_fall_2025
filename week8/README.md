# Materials for week 8

To run the tutorials for this session, the following installation steps must be run, starting from your academy-repo environment:
'''
pip uninstall -y numpy
pip install numpy==1.26.4
conda install -c conda-forge reaktoro=2.12.3 cyipopt=1.4.1
pip install git+https://github.com/watertap-org/reaktoro-pse.git
'''

* Chemistry_Model_Integration_Tutorial.ipynb : this tutorial introduces Reaktoro and the integration of Reaktoro with WaterTAP using Reaktoro-PSE as the connector. The example covers lime and soda ash softening to precipitate calcite.

Post-session folder

* an additional notebook example taken from Reaktoro-PSE examples, which shows how to use Reaktoro-PSE for a softening, acification, reverse osmosis treatment train. Note, the representations of each process unit uses simple Pyomo Blocks instead of WaterTAP unit models in this example. 
