use crate as pallet_iot_store;
use frame_support::derive_impl;
use frame_support::pallet_prelude::ConstU32;

use sp_runtime::BuildStorage;

type Block = frame_system::mocking::MockBlock<Test>;

// Configure a mock runtime to test the pallet.
frame_support::construct_runtime!(
    pub enum Test
    {
        System: frame_system,
        IotStore: pallet_iot_store,
    }
);

#[derive_impl(frame_system::config_preludes::TestDefaultConfig)]
impl frame_system::Config for Test {
    type Block = Block;
}

impl pallet_iot_store::Config for Test {
    type RuntimeEvent = RuntimeEvent;
    type WeightInfo = ();
    type MaxVinLength = ConstU32<32>;
    type MaxManufacturerLength = ConstU32<32>;
    type MaxModelLength = ConstU32<32>;
    type MaxIdentifierLength = ConstU32<32>;
    type MaxVehicles = ConstU32<100000>;
    type VINPrefix = ConstU32<32>;
}

// Build genesis storage according to the mock runtime.
pub fn new_test_ext() -> sp_io::TestExternalities {
    frame_system::GenesisConfig::<Test>::default()
        .build_storage()
        .unwrap()
        .into()
}
