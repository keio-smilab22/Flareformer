from locust import HttpUser, task, constant
import random

image_path_list = ["../flare_transformer/data/magnetogram/2015-07/hmi.M_720s.20150719_105824.magnetogram.png",
                   "../flare_transformer/data/magnetogram/2014-10/hmi.M_720s.20141022_165815.magnetogram.png",
                   "../flare_transformer/data/magnetogram/2014-02/hmi.M_720s.20140213_235811.magnetogram.png",
                   "../flare_transformer/data/magnetogram/2015-01/hmi.M_720s.20150111_125809.magnetogram.png",
                   "../flare_transformer/data/magnetogram/2016-03/hmi.M_720s.20160324_155815.magnetogram.png",
                   "../flare_transformer/data/magnetogram/2016-02/hmi.M_720s.20160212_225810.magnetogram.png",
                   "../flare_transformer/data/magnetogram/2012-07/hmi.M_720s.20120725_235825.magnetogram.png",
                   "../flare_transformer/data/magnetogram/2017-04/hmi.M_720s.20170422_085840.magnetogram.png",
                   "../flare_transformer/data/magnetogram/2011-03/hmi.M_720s.20110312_105815.magnetogram.png",
                   "../flare_transformer/data/magnetogram/2017-09/hmi.M_720s.20170925_175839.magnetogram.png",
                   "../flare_transformer/data/magnetogram/2011-08/hmi.M_720s.20110802_155825.magnetogram.png",
                   "../flare_transformer/data/magnetogram/2017-07/hmi.M_720s.20170716_035846.magnetogram.png",
                   "../flare_transformer/data/magnetogram/2013-09/hmi.M_720s.20130913_075820.magnetogram.png",
                   "../flare_transformer/data/magnetogram/2013-07/hmi.M_720s.20130726_215825.magnetogram.png",
                   "../flare_transformer/data/magnetogram/2012-09/hmi.M_720s.20120916_225820.magnetogram.png",
                   "../flare_transformer/data/magnetogram/2013-12/hmi.M_720s.20131228_195809.magnetogram.png",
                   "../flare_transformer/data/magnetogram/2014-05/hmi.M_720s.20140519_015823.magnetogram.png",
                   "../flare_transformer/data/magnetogram/2012-05/hmi.M_720s.20120527_165825.magnetogram.png",
                   "../flare_transformer/data/magnetogram/2017-05/hmi.M_720s.20170528_015844.magnetogram.png",
                   "../flare_transformer/data/magnetogram/2012-11/hmi.M_720s.20121128_145810.magnetogram.png",
                   "../flare_transformer/data/magnetogram/2012-10/hmi.M_720s.20121002_035818.magnetogram.png",
                   "../flare_transformer/data/magnetogram/2014-09/hmi.M_720s.20140905_195821.magnetogram.png",
                   "../flare_transformer/data/magnetogram/2017-04/hmi.M_720s.20170423_035840.magnetogram.png",
                   "../flare_transformer/data/magnetogram/2011-08/hmi.M_720s.20110827_105823.magnetogram.png",
                   "../flare_transformer/data/magnetogram/2013-05/hmi.M_720s.20130512_125822.magnetogram.png",
                   "../flare_transformer/data/magnetogram/2016-02/hmi.M_720s.20160222_165811.magnetogram.png",
                   "../flare_transformer/data/magnetogram/2014-12/hmi.M_720s.20141206_005810.magnetogram.png",
                   "../flare_transformer/data/magnetogram/2017-09/hmi.M_720s.20170921_055840.magnetogram.png",
                   "../flare_transformer/data/magnetogram/2013-08/hmi.M_720s.20130804_195824.magnetogram.png",
                   "../flare_transformer/data/magnetogram/2010-06/hmi.M_720s.20100626_075826.magnetogram.png",
                   "../flare_transformer/data/magnetogram/2011-05/hmi.M_720s.20110504_065822.magnetogram.png",
                   "../flare_transformer/data/magnetogram/2013-09/hmi.M_720s.20130903_145821.magnetogram.png",
                   "../flare_transformer/data/magnetogram/2011-04/hmi.M_720s.20110416_225820.magnetogram.png",
                   "../flare_transformer/data/magnetogram/2016-06/hmi.M_720s.20160623_095847.magnetogram.png",
                   "../flare_transformer/data/magnetogram/2010-12/hmi.M_720s.20101203_035811.magnetogram.png",
                   "../flare_transformer/data/magnetogram/2010-08/hmi.M_720s.20100816_075824.magnetogram.png",
                   "../flare_transformer/data/magnetogram/2017-12/hmi.M_720s.20171216_195830.magnetogram.png",
                   "../flare_transformer/data/magnetogram/2013-11/hmi.M_720s.20131110_035812.magnetogram.png",
                   "../flare_transformer/data/magnetogram/2014-12/hmi.M_720s.20141217_105809.magnetogram.png",
                   "../flare_transformer/data/magnetogram/2014-09/hmi.M_720s.20140920_135819.magnetogram.png",
                   "../flare_transformer/data/magnetogram/2013-12/hmi.M_720s.20131214_085809.magnetogram.png",
                   "../flare_transformer/data/magnetogram/2015-11/hmi.M_720s.20151125_085810.magnetogram.png",
                   "../flare_transformer/data/magnetogram/2013-03/hmi.M_720s.20130315_225814.magnetogram.png",
                   "../flare_transformer/data/magnetogram/2010-06/hmi.M_720s.20100622_095826.magnetogram.png",
                   "../flare_transformer/data/magnetogram/2012-04/hmi.M_720s.20120423_045821.magnetogram.png",
                   "../flare_transformer/data/magnetogram/2011-01/hmi.M_720s.20110113_045810.magnetogram.png",
                   "../flare_transformer/data/magnetogram/2012-08/hmi.M_720s.20120813_165823.magnetogram.png",
                   "../flare_transformer/data/magnetogram/2013-12/hmi.M_720s.20131201_185810.magnetogram.png",
                   "../flare_transformer/data/magnetogram/2014-03/hmi.M_720s.20140323_115815.magnetogram.png",
                   "../flare_transformer/data/magnetogram/2014-04/hmi.M_720s.20140429_065821.magnetogram.png"]

class WebsiteUser(HttpUser):
    wait_time = constant(60)

    @task
    def test_get_image_bin(self):
        rand_image_path = random.choice(image_path_list)
        with self.client.get(f"/images/bin?path={rand_image_path}", catch_response=True) as response:
            if response.status_code != 200:
                response.failure("statusCode is not 200")
