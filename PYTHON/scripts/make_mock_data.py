from utils.mockdata import generate_mock_data_groups_2

import utils.globalsettings as gs

FEEDER_MARKETS    = ["Spain"]
CUSTOMER_SEGMENTS = ["B2B", "B2C"]
CHANNELS          = ["Booking", "OTAs", "Web"]
HOTEL_SEGMENTS    = ["luxury", "rest"]

def main():
    pars, df = generate_mock_data_groups_2(
        feeder_markets    = FEEDER_MARKETS,
        segmentos_cliente = CUSTOMER_SEGMENTS,
        canales           = CHANNELS,
        segmentos_hotel   = HOTEL_SEGMENTS)

    df.to_csv(gs.the_files.MOCK_DATA, index=False)
    pars.to_csv(gs.the_files.MOCK_PARS, index=False)

if __name__ == "__main__":
    main()