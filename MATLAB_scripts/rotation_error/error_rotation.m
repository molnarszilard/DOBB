% % function [epsilon] = error_rotation(r_estimated,r)
% % %     q_estimated = qGetQ(r_estimated);
% % %     q_inv= qInv(q_estimated);
% % %     q_true = qGetQ(r);
% % %     Q= qMul(q_inv,q_true);
% % %     c = Q(1,:);
% % %     epsilon = 2*acos(abs(c));
% %     
% %     % kvaterniók lekérése
% %     q_estimated = qGetQ(r_estimated);
% %     q_true = qGetQ(r);
% % 
% %     % kvaternió-hiba: q_delta = q_estimated⁻¹ * q_true
% %     q_inv = qInv(q_estimated);
% %     q_delta = qMul(q_inv, q_true);
% % 
% %     % szög kiszámítása a kvaternió skalár részéből (w komponens)
% %     w = q_delta(1);
% %     
% %     % Védelem a határon túli értékek ellen
% %     w = min(max(w, -1), 1);
% % 
% %     % Stabil közelítés kis szögeknél
% %     if (1 - w) < 1e-6
% %         % kis szög: használj sorbővítést: theta ≈ 2 * sqrt(1 - w)
% %         epsilon = 2 * sqrt(2*(1 - w));  % gyorsabb, stabilabb
% %     else
% %         % normál eset
% %         epsilon = 2 * acos(w);
% %     end
% % end

%% Main function
function [epsilon] = error_rotation(R_estimated, R_true)
    % Bemenet: két rotációs mátrix
    % Kimenet: epsilon - forgatási hiba [radiánban]
    
    % Kvaterniók számítása mátrixokból
    q_est = qGetQ(R_estimated);      % q_estimated
    q_true = qGetQ(R_true);          % q_true
    
    % q_err = q_est⁻¹ * q_true
    q_err = qMul(qInv(q_est), q_true);
    
    % Kvaternió skalár része (cos(theta/2))
    c = q_err(1);
    
    % Stabilizált szög kiszámítása:
    epsilon = 2 * acos(min(1.0, max(-1.0, abs(c))));
    
    
%     % Normalizáljuk a hibakvaterniót is a biztonság kedvéért
%     q_err = q_err / norm(q_err);
%    
%     % Skalár és vektor rész szétválasztása
%     q0 = q_err(1);            % skalár rész
%     qv = q_err(2:4);          % vektor rész
% 
%     % Szög kiszámítása Joan Solà szerint (Eq. 45)
%     epsilon = 2 * atan2(norm(qv), q0);


    
end