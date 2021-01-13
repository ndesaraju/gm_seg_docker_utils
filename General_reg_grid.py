def grid_submit(self, shell_cmd, job_name):

                pbsdir = '/data/henry4/jjuwono/PBR_utils/automation/pbs/'

                # define the job name                                                                                
                print('job name', job_name)
                print('shell cmd', shell_cmd)

                # write the shell script that will be qsubbed                                                        
                scriptfile = os.path.join(pbsdir, job_name+'.sh')
                fid = open(scriptfile,"w")
                fid.write("\n".join(["#! /bin/bash",
                                     "#$ -V",
                                     "#$ -q ms.q",
                                     "#$ -l arch=lx24-amd64",
                                     "#$ -v MKL_NUM_THREADS=1",
                                     "#$ -l h_stack=32M",
                                     "#$ -l h_vmem=5G",
                                     "#$ -N {}".format(job_name),
                                     "\n"]))

                fid.write("\n".join(["hostname",
                                     "\n"]))


                #PIPEs the error and output to specific files in the log directory                                   
                fid.write(shell_cmd)
                fid.close()
# write the shell script that will be qsubbed                                                        
                scriptfile = os.path.join(pbsdir, job_name+'.sh')
                fid = open(scriptfile,"w")
                fid.write("\n".join(["#! /bin/bash",
                                     "#$ -V",
                                     "#$ -q ms.q",
                                     "#$ -l arch=lx24-amd64",
                                     "#$ -v MKL_NUM_THREADS=1",
                                     "#$ -l h_stack=32M",
                                     "#$ -l h_vmem=5G",
                                     "#$ -N {}".format(job_name),
                                     "\n"]))

                fid.write("\n".join(["hostname",
                                     "\n"]))


                #PIPEs the error and output to specific files in the log directory                                   
                fid.write(shell_cmd)
                fid.close()

                # Write the qsub command line                                                                        
                qsub = ["cd",pbsdir,";","/netopt/sge_n1ge6/bin/lx24-amd64/qsub", scriptfile]
                cmd = " ".join(qsub)

                # Submit the job                                                                                     
                print("Submitting job {} to grid".format(job_name))
                proc = Popen(cmd,
                             stdout = PIPE,
                             stderr = PIPE,
                             env=os.environ,
                             shell=True,
                             cwd=pbsdir)
                stdout, stderr = proc.communicate()
                job_id = str(stdout).split(' ')[2]

                return job_id


def Syn(self,dim=2,grad_step=0.1,bins=32,convergence='500x500x500',convrg_thresh='1e-5',shrink_factors='1x1x\
1',smoothing_sigmasr='1x1x1',smoothing_sigmas='1x1x1'):
                if not os.path.exists(self.output_path):
                        os.makedirs(self.output_path)
            os.makedirs(self.output_path+'warped')
                if not os.path.exists(self.output_path+'warped1'):
                        os.makedirs(self.output_path+'warped1')

                print(dim)
                #create warped directory in outputh path                                                             

                # instantiate array that will be filled with jobs                                                    
                job_array = []

                for i in range(len(self.files)):

                        # self.static_path is fixed img, self.files[i] is moving img                                 
                        metric_str='MI['+','.join([self.static_path,self.pth+self.files[i],'1','32'])+']'
                        metric_str1='CC['+','.join([self.static_path,self.pth+self.files[i],'1','3'])+']'
                        metric_str2='Mattes['+','.join([self.static_path,self.pth+self.files[i],'1','32'])+']'

                        cmd=' '.join(['antsRegistration','--dimensionality',str(dim),'--output',self.output_path+'wa\
rp'+self.files[i].split('.')[0],'--transform','Rigid['+str(grad_step)+']','--metric',metric_str2,'--convergence','['\
+convergence+','+convrg_thresh+',10]','--shrink-factors',str(shrink_factors),'--smoothing-sigmas',str(smoothing_sigm\
asr),'--transform','Affine['+str(grad_step)+']','--metric',metric_str2,'--convergence','['+convergence+','+convrg_th\
resh+',10]','--shrink-factors',str(shrink_factors),'--smoothing-sigmas',str(smoothing_sigmas),'--transform','SyN[0.1\
,2,1]','--metric',metric_str1,'--convergence','[500x500x500,1e-6,10]','--shrink-factors','1x1x1','--smoothing-sigmas\
','1x1x1','\n'])


                        cmd1=' '.join(['WarpImageMultiTransform',str(dim),self.pth2+self.files2[i],self.output_path+\
'warped/'+self.files2[i].split('.')[0]+'.nii.gz',self.output_path+'warp'+self.files[i].split('.')[0]+'1Warp.nii.gz',\
self.output_path+'warp'+self.files[i].split('.')[0]+'0GenericAffine.mat','-R',self.static_path, '\n'])
                        cmd2=' '.join(['WarpImageMultiTransform',str(dim),self.pth+self.files[i],self.output_path+'w\
arped1/'+self.files[i].split('.')[0]+'.nii.gz',self.output_path+'warp'+self.files[i].split('.')[0]+'1Warp.nii.gz',se\
lf.output_path+'warp'+self.files[i].split('.')[0]+'0GenericAffine.mat','-R',self.static_path, '\n'])

                        grid_job = self.grid_submit( cmd+cmd1+cmd2, '{}_{}_reg'.format(self.static_path.split('/')[-\
1][:4], i))
                        job_array.append(grid_job)

                # check for qstat                                                                                    
                p = Popen(['qstat'], stdout=PIPE)
                qstat = [x.decode("utf-8").split(' ')[0] for x in p.stdout.readlines()[2:]]

                # check to see if no jobs are in qstat                      
                while len([x for x in job_array if x in qstat]) > 0:

                        # sleep the script for 90 seconds                                                            
                        sleep(90)

                        # re-update qstat                                                                            
                        p = Popen(['qstat'], stdout=PIPE)
                        qstat = [x.decode("utf-8").split(' ')[0] for x in p.stdout.readlines()[2:]]
        def Syn_classic(self,dim=2,grad_step=0.1,bins=32,convergence='500x500x500',convrg_thresh='1e-5',shrink_facto\
rs='1x1x1',smoothing_sigmasr='1x1x1',smoothing_sigmas='1x1x1'):
                if not os.path.exists(self.output_path):
                        os.makedirs(self.output_path)
                        os.makedirs(self.output_path+'warped')
                if not os.path.exists(self.output_path+'warped1'):
                        os.makedirs(self.output_path+'warped1')

                print(dim)
                print(self.output_path)

                #create warped directory in outputh path                                                             

                # instantiate array that will be filled with jobs
                job_array = []

                for i in range(len(self.files)):
                        print(self.files[i])
                        # self.static_path is fixed img, self.files[i] is moving img                                 
                        metric_str='MI['+','.join([self.static_path,self.pth+self.files[i],'1','32'])+']'
                        metric_str1='CC['+','.join([self.static_path,self.pth+self.files[i],'1','3'])+']'
                        metric_str2='Mattes['+','.join([self.static_path,self.pth+self.files[i],'1','32'])+']'

                        cmd=['antsRegistration','--dimensionality',str(dim),'--output',self.output_path+'warp'+self.\
files[i].split('.')[0],'-r','['+self.pth+self.files[i]+','+self.static_path+',1]','--transform','Rigid['+str(grad_st\
ep)+']','--metric',metric_str1,'--convergence','['+convergence+','+convrg_thresh+',10]','--shrink-factors',str(shrin\
k_factors),'--smoothing-sigmas',str(smoothing_sigmasr),'--transform','Affine['+str(grad_step)+']','--metric',metric_\
str1,'--convergence','['+convergence+','+convrg_thresh+',10]','--shrink-factors',str(shrink_factors),'--smoothing-si\
gmas',str(smoothing_sigmas),'--transform','SyN[0.1,1,0]','--metric',metric_str1,'--convergence','[500x500x500,1e-6,1\
0]','--shrink-factors','5x3x1','--smoothing-sigmas','1x1x1']

                        #'-r','['+self.static_path+','+self.pth+self.files[i]+',1]'                                  
                        cmd1=['WarpImageMultiTransform',str(dim),self.pth2+self.files2[i],self.output_path+'warped/'\
+self.files2[i].split('.')[0]+'.nii.gz',self.output_path+'warp'+self.files[i].split('.')[0]+'1Warp.nii.gz',self.outp\
ut_path+'warp'+self.files[i].split('.')[0]+'0GenericAffine.mat','-R',self.static_path]
                        cmd2=['WarpImageMultiTransform',str(dim),self.pth+self.files[i],self.output_path+'warped1/'+\
self.files[i].split('.')[0]+'.nii.gz',self.output_path+'warp'+self.files[i].split('.')[0]+'1Warp.nii.gz',self.output\
_path+'warp'+self.files[i].split('.')[0]+'0GenericAffine.mat','-R',self.static_path]

                        process = Popen(cmd, stdout=PIPE)
                        while True:
                                output = process.stdout.readline()
                                print(process.poll())
                                if process.poll() is not None:
                                        break
                                if output:
                                        print(output.strip())
                        #rc = process.poll()                                                                         

                        print('done with reg')
                        #print('done with fixed:{} moving:{}'.format(fixed,moving))                                  


                        p=Popen(cmd1,stdout=PIPE)
