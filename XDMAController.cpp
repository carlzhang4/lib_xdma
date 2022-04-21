#include "XDMAController.h"

#include <cstring>
#include <thread>
#include <chrono>
              
#include <fstream>
#include <iomanip>
#include <immintrin.h>

using namespace std::chrono_literals;
using namespace std;

namespace fpga {

std::mutex XDMAController::ctrl_mutex;
std::mutex XDMAController::btree_mutex;
std::atomic_uint XDMAController::cmdCounter = ATOMIC_VAR_INIT(0);
uint64_t XDMAController::mmTestValue;

XDMAController::XDMAController(int fd, int byfd, int bypass_enable)
{
   //open control device
   m_base = mmap(0, MAP_SIZE, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
   //open bypass device
   if(bypass_enable)
		by_base =  mmap(0, MAP_SIZE_BYPASS, PROT_READ | PROT_WRITE, MAP_SHARED, byfd, 0);
	this->bypass_enable = bypass_enable;
}

XDMAController::~XDMAController()
{
   if (munmap(m_base, MAP_SIZE) == -1)
   {
      std::cerr << "Error on unmap of control device" << std::endl;
   }

	if(bypass_enable){
		if (munmap(by_base, MAP_SIZE_BYPASS) == -1){
			std::cerr << "Error on unmap of bypass device" << std::endl;
		}
	}
}



void XDMAController::writeTlb(unsigned long vaddr, unsigned long paddr, bool isBase,int table_index)
{ 
   std::lock_guard<std::mutex> guard(ctrl_mutex);
   writeReg(8, (uint32_t) vaddr);
   writeReg(9, (uint32_t) (vaddr >> 32));
   writeReg(10, (uint32_t) paddr);
   writeReg(11, (uint32_t) (paddr >> 32));
   writeReg(12, (uint32_t) isBase);
   writeReg(14, (uint32_t) table_index);//
   writeReg(13, (uint32_t) 0);
   writeReg(13, (uint32_t) 1);

}

void XDMAController::setTlb(FpgaPageTable* page_table, int table_index){ 
   for(int i=0;i<page_table->entries;i++){
		writeTlb(page_table->vaddrs[i], page_table->paddrs[i], (i == 0), table_index);
	}
	printf("GDR TLB, index=%d, pages=%ld\n",table_index,page_table->entries);
	// for(int i=0;i<page_table->entries;i++){
	// 	cout<<"v:"<<hex<<page_table->vaddrs[i]/64/1024<<"  p:"<<page_table->paddrs[i]/64/1024<<endl;
	// }
	cout<<"v:"<<hex<<page_table->vaddrs[0]/64/1024<<"  p:"<<page_table->paddrs[0]/64/1024<<endl;
	if(page_table->entries > 1){
		int last_index = page_table->entries-1;
		cout<<"v:"<<hex<<page_table->vaddrs[last_index]/64/1024<<"  p:"<<page_table->paddrs[last_index]/64/1024<<endl;
	}
}

bool XDMAController::checkBypass(){
   return readReg(513);//1 means bypass enable
}


void XDMAController::writeReg(uint32_t addr, uint32_t value){
   volatile uint32_t* wPtr = (uint32_t*) (((uint64_t) m_base) + (uint64_t) ((uint32_t) addr << 2));
   uint32_t writeVal = htols(value);
   *wPtr = writeVal;
}
uint64_t XDMAController::getRegAddr(uint32_t addr){
	volatile uint32_t* wPtr = (uint32_t*) (((uint64_t) m_base) + (uint64_t) ((uint32_t) addr << 2));
	
	return (uint64_t)wPtr;
}
uint64_t XDMAController::getBypassAddr(uint32_t addr){
	volatile __m512i* wPtr =  (__m512i*) (((uint64_t) by_base) + (uint64_t) ((uint32_t) addr << 6));
	// cout<< (uint64_t) by_base<<" "<<(uint64_t)wPtr<<endl;
	return (uint64_t)wPtr;
}

void XDMAController::writeBypassReg(uint32_t addr, uint64_t* value)
{
   if(checkBypass() == 1){
      volatile __m512i* wPtr = (__m512i*) (((uint64_t) by_base) + (uint64_t) ((uint32_t) addr << 6));
      // *wPtr = _mm512_set_epi32 (value[15], value[14], value[13], value[12], value[11], value[10], value[9], value[8], value[7],value[6],value[5],value[4],value[3],value[2],value[1],value[0]);
	//   cout<<(uint64_t)wPtr<<endl;
      *wPtr = _mm512_set_epi64 (value[7],value[6],value[5],value[4],value[3],value[2],value[1],value[0]);
   }else{
      cout<<"bypass disabled, write failed!\n";
   }
}
   



uint32_t XDMAController::readReg(uint32_t addr)
{
   volatile uint32_t* rPtr = (uint32_t*) (((uint64_t) m_base)  + (uint64_t) ((uint32_t) addr << 2));
  return htols(*rPtr);
}

void XDMAController::readBypassReg(uint32_t addr,uint64_t* res)
{
   if(checkBypass() == 1){
      volatile __m512i* rPtr = (__m512i*) (((uint64_t) by_base)  + (uint64_t) ((uint32_t) addr << 6));
      for(int i=0;i<8;i++){
         res[i] = rPtr[0][i];
      }
   }else{
      cout<<"bypass disabled, read failed!\n";
   }
//   return htols(*rPtr);
}






} /* namespace fpga */
