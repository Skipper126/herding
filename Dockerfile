FROM intelopencl/intel-opencl:ubuntu-20.04-ppa
#ENV OCL_INC=/opt/khronos/opencl/include
#ENV OCL_LIB=/opt/intel/opencl-1.2-6.4.0.25/lib64
#ENV ICD_FILE=/etc/OpenCL/vendors/intel.icd
#ENV LD_LIBRARY_PATH $OCL_LIB:$LD_LIBRARY_PATH
#
##COPY --from=opencl $OCL_INC/* $OCL_INC/
#COPY --from=opencl $OCL_LIB/* $OCL_LIB/
#COPY --from=opencl $ICD_FILE $ICD_FILE
#RUN pip3 install pyopencl
CMD clinfo
