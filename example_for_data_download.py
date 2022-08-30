import speasy as spz
ace_mag = spz.get_data('amda/imf', "2016-6-2", "2016-6-5")

amda_tree = spz.inventory.data_tree.amda
ace_mag = spz.get_data(amda_tree.Parameters.ACE.MFI.ace_imf_all.imf, "2016-6-2", "2016-6-5")

sscweb_tree = spz.inventory.data_tree.ssc
solo = spz.get_data(sscweb_tree.Trajectories.solarorbiter, "2021-01-01", "2021-02-01")