#include"XDMACollective.h"
#include"Communicator.cuh"

gdr_mh_t mh_fnic;
gdr_t g_fnic;
int dev_id = 0;
CUdevice dev;
CUcontext dev_ctx;
int n_devices = 0;

fpga::XDMAController* dma_controller;

vector<TLBTable> tables;

extern "C"

using namespace std;
#define LEN_ALIGN(size) (((size)+63) & (~63))

unsigned int* map_reg_4(int reg, fpga::XDMAController* controller){
	cudaError_t err;
	void * addr = (void*)(controller->getRegAddr(reg));
	unsigned int * dev_addr;
	err = cudaHostRegister(addr,4,cudaHostRegisterIoMemory);
	ErrCheck(err);
	cudaHostGetDevicePointer((void **) &(dev_addr), addr, 0);
	return dev_addr;
}

unsigned int* map_reg_64(int reg,fpga::XDMAController* controller){
	cudaError_t err;
	void * addr = (void*)(controller->getBypassAddr(reg));
	unsigned int * dev_addr;
	err = cudaHostRegister(addr,64,cudaHostRegisterIoMemory);
	ErrCheck(err);
	cudaHostGetDevicePointer((void **) &(dev_addr), addr, 0);
	return dev_addr;
}

int gdr_init(){
	ASSERTDRV(cuInit(0));
	ASSERTDRV(cuDeviceGetCount(&n_devices));
	for (int n=0; n<n_devices; ++n) {
		char dev_name[256];
		int dev_pci_domain_id;
		int dev_pci_bus_id;
		int dev_pci_device_id;
		ASSERTDRV(cuDeviceGet(&dev, n));
		ASSERTDRV(cuDeviceGetName(dev_name, sizeof(dev_name) / sizeof(dev_name[0]), dev));
		ASSERTDRV(cuDeviceGetAttribute(&dev_pci_domain_id, CU_DEVICE_ATTRIBUTE_PCI_DOMAIN_ID, dev));
		ASSERTDRV(cuDeviceGetAttribute(&dev_pci_bus_id, CU_DEVICE_ATTRIBUTE_PCI_BUS_ID, dev));
		ASSERTDRV(cuDeviceGetAttribute(&dev_pci_device_id, CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID, dev));
		cout << "GPU id:" << n << "; name: " << dev_name<< "; Bus id: "<< std::hex<< std::setfill('0') << std::setw(4) << dev_pci_domain_id<< ":" << std::setfill('0') << std::setw(2) << dev_pci_bus_id<< ":" << std::setfill('0') << std::setw(2) << dev_pci_device_id<< std::dec<< endl;
		ASSERTDRV(cuDeviceGet(&dev, dev_id));
		ASSERTDRV(cuDevicePrimaryCtxRetain(&dev_ctx, dev));
		ASSERTDRV(cuCtxSetCurrent(dev_ctx));
		g_fnic = gdr_open();
		ASSERT_NEQ(g_fnic, (void*)0);
	}
}

int xdma_init(int machine_rank,int total_machines){
	dma_controller = fpga::XDMA::getController(1,1);

	uint32_t ip_base = 0xc0a8bd00+1;
	uint32_t ipaddr = ip_base + machine_rank;
	uint32_t port = 1234;
	uint32_t conn_ipaddr;
	if(machine_rank==(total_machines-1)){
		conn_ipaddr = ip_base;
	}
	else{
		conn_ipaddr = ipaddr + 1;
	}
	dma_controller->writeReg(0,0);
	dma_controller->writeReg(0,1);
	dma_controller->writeReg(0,0);
	sleep(1);
	dma_controller->writeReg(128,machine_rank);
	dma_controller->writeReg(129,ipaddr);
	dma_controller->writeReg(130,port);
	dma_controller->writeReg(132,conn_ipaddr);
	dma_controller->writeReg(133,port);
	dma_controller->writeReg(131,0);
	dma_controller->writeReg(131,1);
	//tcp listen
	while (((dma_controller->readReg(640)) >> 1) == 0){
		sleep(1);
		cout << "listen status: " << dma_controller->readReg(640) << endl;
		cout << "reg[131] " << dma_controller->readReg(131) << endl;
	};
	cout << "listen status: " << dma_controller->readReg(640) << endl;
	sleep(1);
	//tcp connect
	dma_controller->writeReg(134, (uint32_t)0);
	dma_controller->writeReg(134, (uint32_t)1);
	while (((dma_controller->readReg(641)) >> 16) == 0){
 		cout << "conn status: " << dma_controller->readReg(641) << endl;
		sleep(1);
	};
	unsigned int session_id = dma_controller->readReg(641) & 0x0000ffff;
	cout << "session_id: " << session_id << endl;
	cout << "conn status: " << dma_controller->readReg(641) << endl;
	sleep(1);
	dma_controller->writeReg(134, (uint32_t)0);
   
	int a;
	cout<<"connect success. press any key\n";
	cin>>a;  
	dma_controller -> writeReg(256,session_id);
	dma_controller -> writeReg(257,machine_rank);
	dma_controller -> writeReg(258,(total_machines-1));
	dma_controller -> writeReg(259,32*1024);//boardline of big and small packages
	dma_controller -> writeReg(267,500);//delay cycles

}

int addr_in_tlb(size_t start_page, size_t end_page){
	for(auto it=tables.begin(); it!=tables.end();it++){
		if(it->start_page<=start_page && end_page<=it->end_page){
			return 1;
		}
	}
	return 0;
}

int insert(size_t start_page, size_t end_page){
	int start_contained_index=-1;
	int end_contained_index=-1;
	for(auto it=tables.begin(); it!=tables.end();it++){
		if(it->start_page<=start_page && start_page<=it->end_page){
			start_contained_index = distance(tables.begin(),it);
		}
		if(it->start_page<=end_page && end_page<=it->end_page){
			end_contained_index = distance(tables.begin(),it);
		}
	}
	int last_index=-1;
	for(size_t addr = start_page; addr<=end_page; addr+=GPU_PAGE_SIZE){
		for(auto it=tables.begin(); it!=tables.end();it++){
			if(it->start_page<=addr && addr<=it->end_page){
				int cur_index = distance(tables.begin(),it);
				if(last_index==-1){
					last_index = cur_index;
				}else if(cur_index!=last_index){
					cout<<"Erorr: several mids in different"<<endl;
					ASSERT(0); 
					exit(1);
				}
			}
		}
	}
	TLBTable t;
	cout << "┌------------------------------------┐" ;
	if(start_contained_index==-1 && end_contained_index==-1){
		if(last_index != -1){
			cout<<"Erorr: head and tail not in, but mid in"<<endl;
			ASSERT(0); 
			exit(1);
		}
		cout<<"case <data>\n";
		int pages = (end_page-start_page)/GPU_PAGE_SIZE+1;
		t.start_page = start_page;
		t.end_page = end_page;
		t.tlb.entries = pages;
		t.tlb.vaddrs = (uint64_t*)malloc(sizeof(uint64_t)*pages);
		t.tlb.paddrs = (uint64_t*)malloc(sizeof(uint64_t)*pages);
		do{
			BREAK_IF_NEQ(gdr_pin_buffer(g_fnic, (unsigned long int)start_page, GPU_PAGE_SIZE*pages, 0, 0, &mh_fnic), 0);
		}while(0);
		// printf("page entries:%lu\n",m_page_table.page_entries);

		for(int i=0;i<m_page_table.page_entries;i++){
			t.tlb.paddrs[i]	=	m_page_table.pages[i];
			t.tlb.vaddrs[i]	=	(uint64_t)start_page+i*GPU_PAGE_SIZE;
		}
		int index = tables.size();
		tables.push_back(t);
		dma_controller->setTlb(&(t.tlb),index);
	}else if(start_contained_index!=-1 && end_contained_index==-1){
		auto table_left = tables.at(start_contained_index);
		cout<<"case <old tlb | data>\n";
		size_t real_start_page = table_left.end_page + GPU_PAGE_SIZE;
		int added_pages = (end_page-real_start_page)/GPU_PAGE_SIZE+1;
		int total_pages = table_left.tlb.entries + added_pages;

		t.start_page = table_left.start_page;
		t.end_page = end_page;
		t.tlb.entries = total_pages;
		t.tlb.vaddrs = (uint64_t*)malloc(sizeof(uint64_t)*total_pages);
		t.tlb.paddrs = (uint64_t*)malloc(sizeof(uint64_t)*total_pages);
		do{
			BREAK_IF_NEQ(gdr_pin_buffer(g_fnic, (unsigned long int)real_start_page, GPU_PAGE_SIZE*added_pages, 0, 0, &mh_fnic), 0);
		}while(0);
		printf("original table entries:%lu, gdr_pined pages:%lu\n",table_left.tlb.entries, m_page_table.page_entries);
		ASSERT(m_page_table.page_entries==added_pages);
		for(int i=0;i<table_left.tlb.entries;i++){
			t.tlb.paddrs[i]	=	table_left.tlb.paddrs[i];
			t.tlb.vaddrs[i]	=	table_left.tlb.vaddrs[i];
		}
		int offset = (int)table_left.tlb.entries;
		for(int i=0;i<added_pages;i++){
			t.tlb.paddrs[offset+i]	=	m_page_table.pages[i];
			t.tlb.vaddrs[offset+i]	=	(uint64_t)real_start_page+i*GPU_PAGE_SIZE;
		}
		tables[start_contained_index] = t;
		dma_controller->setTlb(&(t.tlb),start_contained_index);
	}else if(start_contained_index==-1 && end_contained_index!=-1){
		auto table_right = tables.at(end_contained_index);
		cout<<"case <data | old tlb>\n";
		size_t real_end_page = table_right.start_page-GPU_PAGE_SIZE;
		int added_pages = (real_end_page-start_page)/GPU_PAGE_OFFSET+1;
		int total_pages = table_right.tlb.entries + added_pages;

		t.start_page = start_page;
		t.end_page = table_right.end_page;
		t.tlb.entries = total_pages;
		t.tlb.vaddrs = (uint64_t*)malloc(sizeof(uint64_t)*total_pages);
		t.tlb.paddrs = (uint64_t*)malloc(sizeof(uint64_t)*total_pages);
		do{
			BREAK_IF_NEQ(gdr_pin_buffer(g_fnic, (unsigned long int)start_page, GPU_PAGE_SIZE*added_pages, 0, 0, &mh_fnic), 0);
		}while(0);
		printf("original table entries:%lu, gdr_pined pages:%lu\n",table_right.tlb.entries, m_page_table.page_entries);
		ASSERT(m_page_table.page_entries==added_pages);

		for(int i=0;i<added_pages;i++){
			t.tlb.paddrs[i] = m_page_table.pages[i];
			t.tlb.vaddrs[i] = (uint64_t)start_page+i*GPU_PAGE_SIZE;
		}
		for(int i=0;i<table_right.tlb.entries;i++){
			t.tlb.paddrs[added_pages+i]	=	table_right.tlb.paddrs[i];
			t.tlb.vaddrs[added_pages+i]	=	table_right.tlb.vaddrs[i];
		}
		tables[end_contained_index] = t;
		dma_controller->setTlb(&(t.tlb),end_contained_index);
	}

	cout << "└------------------------------------┘"<<endl ;
}

void xdma_all_reduce_gloo(void* send_buf, size_t number, int dtype_size,int rank, int num_machine){
	cout<<"rank:"<<rank<<endl;
	ASSERT(dtype_size==4);
	int len_in_byte = number*dtype_size;
	size_t addr_start = (size_t)send_buf;
	size_t addr_end = addr_start + len_in_byte;

	static unsigned int * fpga_regs[3];

	unsigned int params[8];

	unsigned int token_divide = 11;
	unsigned int token_mul = 3;
	ONCE_RUN(gdr_init();
			xdma_init(rank,num_machine);
			dma_controller -> writeReg(265,token_divide);
			dma_controller -> writeReg(266,token_mul);

			fpga_regs[0] = map_reg_4(797,dma_controller);//response
			fpga_regs[1] = map_reg_64(8,dma_controller);//bypass
			cout<<"num_machine:"<<num_machine<<" "<<"rank:"<<rank<<endl;
			);

	{
		size_t start_page	= (addr_start& GPU_PAGE_MASK);
		size_t end_page		= ((addr_end-1)& GPU_PAGE_MASK);
		int pages = (end_page-start_page)/GPU_PAGE_SIZE+1;
		if(!addr_in_tlb(start_page, end_page)){	
			insert(start_page,end_page);
		}else{
			// cout<<hex<<addr_start<<" "<<dec<<number<<endl;
		}
	}
	size_t addr_start_64B = addr_start & 0xFFFFFFFFFFFFFFC0;
	size_t addr_end_64B = (addr_end + 64 - 1)& 0xFFFFFFFFFFFFFFC0;
	unsigned int len_out = (int)(addr_end_64B-addr_start_64B);

	unsigned int head_length = addr_start - addr_start_64B;
	unsigned int each_client_length = LEN_ALIGN((head_length + len_in_byte - 1)/num_machine+1);
	unsigned int tail_length = each_client_length*num_machine - head_length - len_in_byte;
	unsigned int total_length = each_client_length*num_machine;
	
	params[0] = total_length;
	params[1] = each_client_length;
	params[2] = (unsigned int)addr_start_64B;
	params[3] = (unsigned int)(addr_start_64B>>32);
	params[4] = head_length;
	params[5] = tail_length;

	cu_all_reduce_float_gloo((float*)addr_start, len_in_byte, fpga_regs, params);
}

void xdma_all_reduce_test(void* send_buf, void* recv_buf, size_t number, int dtype, cudaStream_t stream,int rank, int num_machine){
	// int xin;
	// cin>>xin;
	// cout<<xin<<endl;

	ASSERT(send_buf==recv_buf);

	int dtype_size=4;

	int len_in_byte = number*dtype_size;
	size_t addr_start = (size_t)send_buf;
	size_t addr_end = addr_start + len_in_byte;

	// cout<<hex<<addr_start<<" "<<dec<<len_in_byte<<endl;


	static unsigned int * fpga_regs[3];

	unsigned int params[8];

	unsigned int token_divide = 9;
	unsigned int token_mul = 3;
	ONCE_RUN(gdr_init();
			xdma_init(rank,num_machine);
			dma_controller -> writeReg(265,token_divide);
			dma_controller -> writeReg(266,token_mul);

			fpga_regs[0] = map_reg_4(797,dma_controller);//response
			fpga_regs[1] = map_reg_64(8,dma_controller);//bypass
			fpga_regs[2] = map_reg_4(649,dma_controller);//

			cout<<"num_machine:"<<num_machine<<" "<<"rank:"<<rank<<endl;
			);

	{
		size_t start_page	= (addr_start& GPU_PAGE_MASK);
		size_t end_page		= ((addr_end-1)& GPU_PAGE_MASK);
		int pages = (end_page-start_page)/GPU_PAGE_SIZE+1;
		if(!addr_in_tlb(start_page, end_page)){	
			insert(start_page,end_page);
		}else{
			// cout<<hex<<addr_start<<" "<<dec<<number<<endl;
		}
	}
	size_t addr_start_64B = addr_start & 0xFFFFFFFFFFFFFFC0;
	size_t addr_end_64B = (addr_end + 64 - 1)& 0xFFFFFFFFFFFFFFC0;
	unsigned int len_out = (int)(addr_end_64B-addr_start_64B);

	unsigned int head_length = addr_start - addr_start_64B;
	unsigned int each_client_length = LEN_ALIGN((head_length + len_in_byte - 1)/num_machine+1);
	unsigned int tail_length = each_client_length*num_machine - head_length - len_in_byte;
	unsigned int total_length = each_client_length*num_machine;
	
	params[0] = total_length;
	params[1] = each_client_length;
	params[2] = (unsigned int)addr_start_64B;
	params[3] = (unsigned int)(addr_start_64B>>32);
	params[4] = head_length;
	params[5] = tail_length;
	params[6] = dtype;

	cu_all_reduce_float((float*)addr_start, len_in_byte, fpga_regs, params, stream);
}

// int xdma_all_reduce(size_t addr_start,size_t number,int size){
// 	size_t len_in_byte = number*size;
// 	size_t addr_end = addr_start + len_in_byte;
// 	static int count=50;
// 	if(count>0){
// 		count--;
// 	}else{
// 		return 0;
// 	}
// 	static int first_touch=1;
// 	if(first_touch){
// 		first_touch=0;
// 		gdr_init();
// 		// xdma_init();
// 		// float p_cpu[2000];
// 		// cudaMemcpy(p_cpu,p,length,cudaMemcpyDeviceToHost);
// 		// for(int i=0;i<number;i++){
// 		// 	printf("%f ",p_cpu[i]);
// 		// }
// 	}
// 	{
// 		size_t start_page	= (addr_start& GPU_PAGE_MASK);
// 		size_t end_page		= ((addr_end-1)& GPU_PAGE_MASK);
// 		int pages = (end_page-start_page)/GPU_PAGE_SIZE+1;
// 		if(!addr_in_tlb(start_page, end_page)){	
// 			insert(start_page,end_page);
// 		}else{
// 			// cout<<"addr in tlb\n";
// 		}
// 		// int x;
// 		// cin>>x;
// 		// cout<<x<<endl;

// 		float *p = (float*)addr_start;
// 		size_t addr_start_64B = addr_start & 0xFFFFFFFFFFFFFFC0;
// 		size_t addr_end_64B = (addr_end + 64 - 1)& 0xFFFFFFFFFFFFFFC0;
// 		unsigned int len_out = (int)(addr_end_64B-addr_start_64B);

// 		cout<<"origin start: \t"<<hex<<addr_start<<		"  origin end: "<<addr_end<<"  origin len: "<<len_in_byte<<endl;
// 		// cout<<"start page:   \t"<<hex<<start_page/GPU_PAGE_SIZE<<"      end page:   "<<end_page/GPU_PAGE_SIZE<<"  pages:"<<dec<<pages<<endl;
// 		// cout<<"read addr:    \t"<<hex<<addr_start_64B<<	"  len_out:    "<<hex<<len_out<<endl;

// 		// unsigned int original_cmd = dma_controller->readReg(525);
// 		// unsigned int original_cmd1 = dma_controller->readReg(536);

// 		// unsigned int original_word = dma_controller->readReg(526);

// 		// unsigned int head_length = addr_start - addr_start_64B;

		
// 		// unsigned int each_client_length = LEN_ALIGN((head_length + len_in_byte)/num+1);
// 		// unsigned int tail_length = each_client_length*num - head_length - len_in_byte;
// 		// unsigned int token_divide = 11;
// 		// unsigned int token_mul = 3;

// 		// dma_controller -> writeReg(259,each_client_length);
// 		// dma_controller -> writeReg(261,(unsigned int)addr_start_64B);
// 		// dma_controller -> writeReg(262,(unsigned int)(addr_start_64B>>32));
// 		// dma_controller -> writeReg(263,head_length);
// 		// dma_controller -> writeReg(264,tail_length);
// 		// dma_controller -> writeReg(265,token_divide);
// 		// dma_controller -> writeReg(266,token_mul);
// 		// dma_controller -> writeReg(260,0);
// 		// dma_controller -> writeReg(260,1);//start
// 		// sleep(1);   
// 		// cout << "write_cmd0_counter: " << dma_controller->readReg(768) << endl;
// 		// cout << "write_data0_counter: " << dma_controller->readReg(769) << endl;
// 		// cout << "read_cmd0_counter: " << dma_controller->readReg(772) << endl;
// 		// cout << "read_data0_counter: " << dma_controller->readReg(773) << endl;
// 		// cout << "dma_wr_cmd0_counter: " << dma_controller->readReg(776) << endl;
// 		// cout << "dma_wr_data0_counter: " << dma_controller->readReg(777) << endl;
// 		// cout << "dma_rd_cmd1_counter: " << dma_controller->readReg(778) << endl;
// 		// cout << "dma_rd_data1_counter: " << dma_controller->readReg(779) << endl;
// 		// cout << "tcp_tx_cmd0_counter: " << dma_controller->readReg(788) << endl;
// 		// cout << "tcp_tx_data0_counter: " << dma_controller->readReg(789) << endl;
// 		// cout << "tcp_rx_cmd0_counter: " << dma_controller->readReg(790) << endl;
// 		// cout << "tcp_rx_data0_counter: " << dma_controller->readReg(791) << endl;

// 		// cout << "rx meta over flow cnt: " << dma_controller->readReg(648) << endl;
// 		// cout << "rx data over flow cnt: " << dma_controller->readReg(649) << endl;

// 		// cout << "time_counter: " << dma_controller->readReg(792) << endl;
// 		// cout << "time_counter: " << dma_controller->readReg(792) << endl;
// 		// cout<<hex<<"delta cmd0: "	<<dma_controller ->readReg(525) - original_cmd<<endl;
// 		// cout<<hex<<"delta cmd1: "	<<dma_controller ->readReg(536) - original_cmd1<<endl;
// 		// cout<<hex<<"delta byte: "	<<(dma_controller ->readReg(526) - original_word)*64<<endl;
// 		// cout<<endl;
// 		// cout<<hex<<"pkg:"	<<dma_controller ->	readReg(527)<<endl;
// 		// cout<<hex<<"len:"	<<dma_controller ->	readReg(528)<<endl;
// 		// cout<<hex<<"reg[520]: 0x"	<<dma_controller ->	readReg(520)<<endl;
// 		// cout<<hex<<"reg[521]: 0x"	<<dma_controller ->	readReg(521)<<endl;
// 		// cout<<hex<<"reg[531]: 0x"	<<dma_controller ->	readReg(531)<<endl;
// 		// cout<<hex<<"reg[532]: 0x"	<<dma_controller ->	readReg(532)<<endl;
// 	}
// }
