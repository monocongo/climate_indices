import logging
import numpy as np
import unittest

from scripts import netcdf_utils

#-----------------------------------------------------------------------------------------------------------------------
# disable logging messages
logging.disable(logging.CRITICAL)

#-------------------------------------------------------------------------------------------------------------------------------------------
class NetcdfUtilsTestCase(unittest.TestCase):
    '''
    Tests for `netcdf_utils.py`.
    '''

    #----------------------------------------------------------------------------------------
    def test_find_netcdf_datatype(self):
        '''
        Test for the netcdf_utils.find_netcdf_datatype() function
        '''

        data_objs = (['f8', float(1.0)], 
                     ['f4', np.float32(1.0)], 
                     ['f8', np.float64(1.0)], 
                     ['f8', np.double(1.0)],
                     ['f8', np.NaN],
                     ['i2', np.int16(1.0)],
                     ['i4', np.int32(1.0)],
                     ['i4', int(1.0)])
        for [data_type, data_object] in data_objs:
                     
            self.assertEqual(data_type, 
                             netcdf_utils.find_netcdf_datatype(data_object), 
                             'Failed to determine expected data type for {0}, expected {1}'.format(data_object, 
                                                                                                   data_type))
         
        # provide nonsense arguments, should raise error in each case
        with self.assertRaises(ValueError):
            netcdf_utils.find_netcdf_datatype(data_object={'abc': 123})
            netcdf_utils.find_netcdf_datatype("hokey pokey")
            netcdf_utils.find_netcdf_datatype(['this is a list of heterogeneous items', 14, {43: 56}])
        
#--------------------------------------------------------------------------------------------
if __name__ == '__main__':
    unittest.main()
    