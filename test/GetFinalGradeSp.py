def GetFinalGrade(Final_list):

    id, m_len, mid_len, m_wid, mid_wid, f_grad=Final_list
    if mid_len<9 and mid_len>=6 and mid_wid <3.5 and (f_grad=="Normal" or f_grad=="Minor defects"):
        F_SampleGrade="Premium"
    elif mid_len<6 and mid_len>=3 and mid_wid <3.5 and (f_grad=="Normal" or f_grad=="Minor defects"):
        F_SampleGrade = "Good"
    elif mid_wid <3.5 and f_grad=="Severe defects":
        F_SampleGrade = "Fair"
    else:
        F_SampleGrade = "Cull"
    return F_SampleGrade

if __name__=="__main__":
    Final_list=[1,15,4.5,15,15,"Normal"]
    F_SampleGrade=GetFinalGrade(Final_list)
    print(F_SampleGrade)
